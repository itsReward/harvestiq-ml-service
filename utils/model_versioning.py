"""
Model Versioning and Management Utilities
Handles model storage, versioning, and deployment management
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib
import zipfile

from models.ensemble_predictor import EnsembleYieldPredictor

logger = logging.getLogger(__name__)

class ModelVersionManager:
    """
    Manages model versions, storage, and deployment
    """

    def __init__(self, base_path: str = "./models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.versions_path = self.base_path / "versions"
        self.versions_path.mkdir(parents=True, exist_ok=True)

        self.active_path = self.base_path / "active"
        self.active_path.mkdir(parents=True, exist_ok=True)

        self.archive_path = self.base_path / "archive"
        self.archive_path.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.base_path / "version_metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load version metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {str(e)}")

        return {
            "versions": {},
            "active_version": None,
            "last_updated": None
        }

    def _save_metadata(self) -> None:
        """Save version metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")

    def save_new_version(self, model: EnsembleYieldPredictor,
                        version_tag: Optional[str] = None,
                        description: Optional[str] = None) -> Path:
        """
        Save a new model version
        """
        # Generate version identifier
        if version_tag is None:
            version_tag = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create version directory
        version_path = self.versions_path / version_tag
        version_path.mkdir(parents=True, exist_ok=True)

        try:
            # Save the model
            model.save_model(version_path)

            # Calculate model hash for integrity verification
            model_hash = self._calculate_model_hash(version_path)

            # Create version metadata
            version_metadata = {
                "version_tag": version_tag,
                "created_at": datetime.now().isoformat(),
                "description": description or f"Model version {version_tag}",
                "model_version": model.get_version(),
                "training_date": model.get_training_date(),
                "metrics": model.get_metrics(),
                "feature_count": model.get_feature_count(),
                "model_hash": model_hash,
                "status": "available",
                "deployment_count": 0,
                "path": str(version_path.relative_to(self.base_path))
            }

            # Save version-specific metadata
            with open(version_path / "version_info.json", 'w') as f:
                json.dump(version_metadata, f, indent=2, default=str)

            # Update global metadata
            self.metadata["versions"][version_tag] = version_metadata
            self.metadata["last_updated"] = datetime.now().isoformat()
            self._save_metadata()

            logger.info(f"Model version {version_tag} saved successfully")
            return version_path

        except Exception as e:
            logger.error(f"Error saving model version {version_tag}: {str(e)}")
            # Cleanup on failure
            if version_path.exists():
                shutil.rmtree(version_path)
            raise

    def _calculate_model_hash(self, model_path: Path) -> str:
        """Calculate hash of model files for integrity verification"""
        hasher = hashlib.sha256()

        # Hash all model files
        for file_path in sorted(model_path.glob("**/*")):
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)

        return hasher.hexdigest()

    def get_latest_model_path(self) -> Optional[Path]:
        """Get path to the latest model version"""
        if not self.metadata["versions"]:
            return None

        # Sort versions by creation time
        latest_version = max(
            self.metadata["versions"].values(),
            key=lambda x: x["created_at"]
        )

        version_path = self.base_path / latest_version["path"]
        if version_path.exists():
            return version_path

        logger.warning(f"Latest model path not found: {version_path}")
        return None

    def get_version_path(self, version_tag: str) -> Path:
        """Get path to a specific model version"""
        if version_tag not in self.metadata["versions"]:
            raise ValueError(f"Version {version_tag} not found")

        version_info = self.metadata["versions"][version_tag]
        return self.base_path / version_info["path"]

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all available model versions"""
        versions = []

        for version_tag, version_info in self.metadata["versions"].items():
            # Check if version still exists
            version_path = self.base_path / version_info["path"]
            if version_path.exists():
                versions.append({
                    "version_tag": version_tag,
                    "created_at": version_info["created_at"],
                    "description": version_info.get("description", ""),
                    "metrics": version_info.get("metrics", {}),
                    "status": version_info.get("status", "unknown"),
                    "deployment_count": version_info.get("deployment_count", 0)
                })

        # Sort by creation time (newest first)
        versions.sort(key=lambda x: x["created_at"], reverse=True)
        return versions

    def activate_version(self, version_tag: str) -> bool:
        """Activate a specific model version"""
        if version_tag not in self.metadata["versions"]:
            logger.error(f"Version {version_tag} not found")
            return False

        try:
            version_path = self.get_version_path(version_tag)

            if not version_path.exists():
                logger.error(f"Version {version_tag} files not found at {version_path}")
                return False

            # Verify model integrity
            if not self._verify_model_integrity(version_tag):
                logger.error(f"Model integrity check failed for version {version_tag}")
                return False

            # Clear current active model
            if self.active_path.exists():
                shutil.rmtree(self.active_path)
            self.active_path.mkdir(parents=True, exist_ok=True)

            # Copy version to active directory
            shutil.copytree(version_path, self.active_path / "current", dirs_exist_ok=True)

            # Update metadata
            old_active = self.metadata.get("active_version")
            if old_active and old_active in self.metadata["versions"]:
                self.metadata["versions"][old_active]["deployment_count"] = \
                    self.metadata["versions"][old_active].get("deployment_count", 0) + 1

            self.metadata["active_version"] = version_tag
            self.metadata["versions"][version_tag]["status"] = "active"
            self.metadata["versions"][version_tag]["last_activated"] = datetime.now().isoformat()

            # Mark other versions as inactive
            for v_tag, v_info in self.metadata["versions"].items():
                if v_tag != version_tag and v_info.get("status") == "active":
                    v_info["status"] = "available"

            self._save_metadata()

            logger.info(f"Model version {version_tag} activated successfully")
            return True

        except Exception as e:
            logger.error(f"Error activating version {version_tag}: {str(e)}")
            return False

    def _verify_model_integrity(self, version_tag: str) -> bool:
        """Verify model file integrity"""
        try:
            version_info = self.metadata["versions"][version_tag]
            stored_hash = version_info.get("model_hash")

            if not stored_hash:
                logger.warning(f"No stored hash for version {version_tag}")
                return True  # Allow if no hash stored

            version_path = self.get_version_path(version_tag)
            current_hash = self._calculate_model_hash(version_path)

            if stored_hash != current_hash:
                logger.error(f"Hash mismatch for version {version_tag}: {stored_hash} != {current_hash}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error verifying integrity for version {version_tag}: {str(e)}")
            return False

    def delete_version(self, version_tag: str, force: bool = False) -> bool:
        """Delete a model version"""
        if version_tag not in self.metadata["versions"]:
            logger.error(f"Version {version_tag} not found")
            return False

        # Prevent deletion of active version unless forced
        if (self.metadata.get("active_version") == version_tag and not force):
            logger.error(f"Cannot delete active version {version_tag} without force=True")
            return False

        try:
            version_path = self.get_version_path(version_tag)

            # Archive before deletion
            if self._archive_version(version_tag):
                logger.info(f"Version {version_tag} archived before deletion")

            # Remove files
            if version_path.exists():
                shutil.rmtree(version_path)

            # Update metadata
            del self.metadata["versions"][version_tag]

            if self.metadata.get("active_version") == version_tag:
                self.metadata["active_version"] = None

            self._save_metadata()

            logger.info(f"Model version {version_tag} deleted successfully")
            return True

        except Exception as e:
            logger.error(f"Error deleting version {version_tag}: {str(e)}")
            return False

    def _archive_version(self, version_tag: str) -> bool:
        """Archive a model version before deletion"""
        try:
            version_path = self.get_version_path(version_tag)

            if not version_path.exists():
                return False

            # Create archive file
            archive_file = self.archive_path / f"{version_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

            with zipfile.ZipFile(archive_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in version_path.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(version_path)
                        zipf.write(file_path, arcname)

            logger.info(f"Version {version_tag} archived to {archive_file}")
            return True

        except Exception as e:
            logger.error(f"Error archiving version {version_tag}: {str(e)}")
            return False

    def restore_from_archive(self, archive_file: str, version_tag: Optional[str] = None) -> bool:
        """Restore a model version from archive"""
        try:
            archive_path = Path(archive_file)
            if not archive_path.exists():
                logger.error(f"Archive file not found: {archive_file}")
                return False

            # Generate version tag if not provided
            if version_tag is None:
                version_tag = f"restored_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Create restoration directory
            restore_path = self.versions_path / version_tag
            restore_path.mkdir(parents=True, exist_ok=True)

            # Extract archive
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                zipf.extractall(restore_path)

            # Load version metadata if available
            version_info_file = restore_path / "version_info.json"
            if version_info_file.exists():
                with open(version_info_file, 'r') as f:
                    version_info = json.load(f)

                # Update paths and timestamps
                version_info["version_tag"] = version_tag
                version_info["restored_at"] = datetime.now().isoformat()
                version_info["status"] = "restored"
                version_info["path"] = str(restore_path.relative_to(self.base_path))

                # Update global metadata
                self.metadata["versions"][version_tag] = version_info
                self._save_metadata()

            logger.info(f"Model version restored as {version_tag}")
            return True

        except Exception as e:
            logger.error(f"Error restoring from archive {archive_file}: {str(e)}")
            return False

    def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two model versions"""
        if version1 not in self.metadata["versions"]:
            raise ValueError(f"Version {version1} not found")

        if version2 not in self.metadata["versions"]:
            raise ValueError(f"Version {version2} not found")

        v1_info = self.metadata["versions"][version1]
        v2_info = self.metadata["versions"][version2]

        comparison = {
            "version1": {
                "tag": version1,
                "created_at": v1_info["created_at"],
                "metrics": v1_info.get("metrics", {}),
                "feature_count": v1_info.get("feature_count", 0)
            },
            "version2": {
                "tag": version2,
                "created_at": v2_info["created_at"],
                "metrics": v2_info.get("metrics", {}),
                "feature_count": v2_info.get("feature_count", 0)
            },
            "comparison": {}
        }

        # Compare metrics
        v1_metrics = v1_info.get("metrics", {})
        v2_metrics = v2_info.get("metrics", {})

        for metric in set(v1_metrics.keys()) | set(v2_metrics.keys()):
            v1_val = v1_metrics.get(metric, 0)
            v2_val = v2_metrics.get(metric, 0)

            comparison["comparison"][metric] = {
                "version1": v1_val,
                "version2": v2_val,
                "difference": v2_val - v1_val,
                "improvement": v2_val > v1_val if metric in ["r2_score", "accuracy_within_10_percent"] else v2_val < v1_val
            }

        # Overall recommendation
        key_metrics = ["test_r2", "test_mae", "test_rmse"]
        improvements = 0
        total_key_metrics = 0

        for metric in key_metrics:
            if metric in comparison["comparison"]:
                total_key_metrics += 1
                if comparison["comparison"][metric]["improvement"]:
                    improvements += 1

        if total_key_metrics > 0:
            improvement_ratio = improvements / total_key_metrics
            if improvement_ratio >= 0.67:
                comparison["recommendation"] = f"Version {version2} shows significant improvement"
            elif improvement_ratio >= 0.33:
                comparison["recommendation"] = "Mixed results - manual review recommended"
            else:
                comparison["recommendation"] = f"Version {version1} appears to perform better"
        else:
            comparison["recommendation"] = "Insufficient metrics for comparison"

        return comparison

    def cleanup_old_versions(self, keep_latest: int = 5, keep_days: int = 30) -> List[str]:
        """Clean up old model versions"""
        deleted_versions = []

        try:
            versions = list(self.metadata["versions"].items())
            versions.sort(key=lambda x: x[1]["created_at"], reverse=True)

            cutoff_date = datetime.now() - timedelta(days=keep_days)
            active_version = self.metadata.get("active_version")

            for i, (version_tag, version_info) in enumerate(versions):
                # Skip if it's the active version
                if version_tag == active_version:
                    continue

                # Skip if within keep_latest count
                if i < keep_latest:
                    continue

                # Check if older than cutoff
                created_at = datetime.fromisoformat(version_info["created_at"])
                if created_at < cutoff_date:
                    if self.delete_version(version_tag):
                        deleted_versions.append(version_tag)

            if deleted_versions:
                logger.info(f"Cleaned up {len(deleted_versions)} old versions: {deleted_versions}")

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

        return deleted_versions

    def export_version_info(self, output_file: str) -> bool:
        """Export version information to file"""
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_versions": len(self.metadata["versions"]),
                "active_version": self.metadata.get("active_version"),
                "versions": []
            }

            for version_tag, version_info in self.metadata["versions"].items():
                export_data["versions"].append({
                    "version_tag": version_tag,
                    "created_at": version_info["created_at"],
                    "description": version_info.get("description", ""),
                    "metrics": version_info.get("metrics", {}),
                    "status": version_info.get("status", "unknown"),
                    "deployment_count": version_info.get("deployment_count", 0),
                    "feature_count": version_info.get("feature_count", 0)
                })

            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Version information exported to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Error exporting version info: {str(e)}")
            return False

    def get_version_performance_history(self) -> Dict[str, Any]:
        """Get performance history across all versions"""
        history = {
            "timestamps": [],
            "metrics": {},
            "versions": []
        }

        # Sort versions by creation time
        versions = list(self.metadata["versions"].items())
        versions.sort(key=lambda x: x[1]["created_at"])

        for version_tag, version_info in versions:
            created_at = version_info["created_at"]
            metrics = version_info.get("metrics", {})

            history["timestamps"].append(created_at)
            history["versions"].append(version_tag)

            # Collect metrics
            for metric_name, metric_value in metrics.items():
                if metric_name not in history["metrics"]:
                    history["metrics"][metric_name] = []
                history["metrics"][metric_name].append(metric_value)

        return history

    def validate_version(self, version_tag: str) -> Dict[str, Any]:
        """Validate a specific model version"""
        validation_result = {
            "version_tag": version_tag,
            "is_valid": False,
            "issues": [],
            "warnings": []
        }

        try:
            if version_tag not in self.metadata["versions"]:
                validation_result["issues"].append("Version not found in metadata")
                return validation_result

            version_info = self.metadata["versions"][version_tag]
            version_path = self.get_version_path(version_tag)

            # Check if files exist
            if not version_path.exists():
                validation_result["issues"].append("Version directory not found")
                return validation_result

            # Check required files
            required_files = ["sklearn_models.joblib", "meta_learner.joblib",
                            "scalers.joblib", "metadata.json"]

            for required_file in required_files:
                if not (version_path / required_file).exists():
                    validation_result["issues"].append(f"Missing required file: {required_file}")

            # Check model integrity
            if not self._verify_model_integrity(version_tag):
                validation_result["issues"].append("Model integrity check failed")

            # Check metrics
            metrics = version_info.get("metrics", {})
            if not metrics:
                validation_result["warnings"].append("No performance metrics available")
            else:
                # Check for reasonable metric values
                if "test_r2" in metrics:
                    r2_score = metrics["test_r2"]
                    if r2_score < 0.5:
                        validation_result["warnings"].append(f"Low R² score: {r2_score:.3f}")
                    elif r2_score > 0.95:
                        validation_result["warnings"].append(f"Suspiciously high R² score: {r2_score:.3f}")

            # Check age
            created_at = datetime.fromisoformat(version_info["created_at"])
            age_days = (datetime.now() - created_at).days

            if age_days > 90:
                validation_result["warnings"].append(f"Model is {age_days} days old")

            # If no critical issues, mark as valid
            if not validation_result["issues"]:
                validation_result["is_valid"] = True

        except Exception as e:
            validation_result["issues"].append(f"Validation error: {str(e)}")

        return validation_result

    def get_active_model_path(self) -> Optional[Path]:
        """Get path to the currently active model"""
        active_model_path = self.active_path / "current"

        if active_model_path.exists():
            return active_model_path

        # Fallback to latest version if no active model
        return self.get_latest_model_path()

    def get_version_metadata(self, version_tag: str) -> Dict[str, Any]:
        """Get detailed metadata for a specific version"""
        if version_tag not in self.metadata["versions"]:
            raise ValueError(f"Version {version_tag} not found")

        return self.metadata["versions"][version_tag].copy()

    def update_version_description(self, version_tag: str, description: str) -> bool:
        """Update the description for a model version"""
        if version_tag not in self.metadata["versions"]:
            logger.error(f"Version {version_tag} not found")
            return False

        try:
            self.metadata["versions"][version_tag]["description"] = description
            self.metadata["versions"][version_tag]["last_modified"] = datetime.now().isoformat()
            self._save_metadata()

            # Also update version-specific metadata file
            version_path = self.get_version_path(version_tag)
            version_info_file = version_path / "version_info.json"

            if version_info_file.exists():
                with open(version_info_file, 'r') as f:
                    version_info = json.load(f)

                version_info["description"] = description
                version_info["last_modified"] = datetime.now().isoformat()

                with open(version_info_file, 'w') as f:
                    json.dump(version_info, f, indent=2, default=str)

            logger.info(f"Updated description for version {version_tag}")
            return True

        except Exception as e:
            logger.error(f"Error updating description for version {version_tag}: {str(e)}")
            return False