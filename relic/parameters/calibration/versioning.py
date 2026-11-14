"""
Parameter Version Control System

This module provides comprehensive version control and provenance tracking
for parameter calibration results. It enables tracking of parameter evolution,
calibration history, and facilitates reproducibility in scientific modeling.

Scientific Background:
    Parameter versioning is crucial for scientific reproducibility and 
    model development. It allows researchers to:
    - Track parameter evolution over time
    - Compare different calibration runs
    - Rollback to previous parameter sets
    - Maintain provenance of calibration results

References:
    - Peng (2011). Reproducible research in computational science.
    - FAIR data principles for scientific parameter management
    - Git version control concepts adapted for parameter management
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
import json
import hashlib
import uuid
from pathlib import Path
from enum import Enum
import logging
from contextlib import contextmanager
import threading

from .results import CalibrationResult
from ...core.managers.base import BaseManager, ManagerState
from ...core.managers.metadata import ManagerMetadata

logger = logging.getLogger(__name__)


class ParameterChangeType(Enum):
    """Types of parameter changes"""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    CALIBRATED = "calibrated"
    VALIDATED = "validated"
    ROLLED_BACK = "rolled_back"
    MERGED = "merged"
    BRANCHED = "branched"


@dataclass
class ParameterChange:
    """Record of a single parameter change"""
    parameter_name: str
    change_type: ParameterChangeType
    old_value: Optional[float] = None
    new_value: Optional[float] = None
    timestamp: Optional[datetime] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['change_type'] = self.change_type.value
        data['timestamp'] = self.timestamp.isoformat() if self.timestamp else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterChange':
        """Create from dictionary"""
        data = data.copy()
        data['change_type'] = ParameterChangeType(data['change_type'])
        if data['timestamp']:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class VersionMetadata:
    """Metadata for a parameter version"""
    version_id: str
    branch_name: str  # branch the version was created on (record field, no default)
    parent_version_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    author: Optional[str] = None
    message: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    calibration_result: Optional[CalibrationResult] = None
    changes: List[ParameterChange] = field(default_factory=list)
    checksum: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.version_id is None:
            self.version_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat() if self.timestamp else None
        data['changes'] = [change.to_dict() for change in self.changes]
        if self.calibration_result:
            data['calibration_result'] = self.calibration_result.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VersionMetadata':
        """Create from dictionary"""
        data = data.copy()
        if data['timestamp']:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['changes'] = [ParameterChange.from_dict(change) for change in data.get('changes', [])]
        if data.get('calibration_result'):
            data['calibration_result'] = CalibrationResult.from_dict(data['calibration_result'])
        return cls(**data)


@dataclass
class ParameterVersion:
    """Complete parameter version including values and metadata"""
    parameters: Dict[str, float]
    metadata: VersionMetadata
    
    def compute_checksum(self) -> str:
        """Compute checksum of parameter values"""
        # Sort parameters for consistent checksum
        sorted_params = dict(sorted(self.parameters.items()))
        param_str = json.dumps(sorted_params, sort_keys=True)
        return hashlib.sha256(param_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'parameters': self.parameters,
            'metadata': self.metadata.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterVersion':
        """Create from dictionary"""
        return cls(
            parameters=data['parameters'],
            metadata=VersionMetadata.from_dict(data['metadata'])
        )


class ParameterDiff:
    """Compute and represent differences between parameter versions"""
    
    @staticmethod
    def compute_diff(version1: ParameterVersion, 
                    version2: ParameterVersion) -> Dict[str, Dict[str, Any]]:
        """
        Compute difference between two parameter versions
        
        Parameters:
            version1: First version (typically older)
            version2: Second version (typically newer)
            
        Returns:
            Dictionary with parameter differences
        """
        params1 = version1.parameters
        params2 = version2.parameters
        
        diff = {}
        
        # Find all parameter names
        all_params = set(params1.keys()) | set(params2.keys())
        
        for param in all_params:
            val1 = params1.get(param)
            val2 = params2.get(param)
            
            if val1 is None and val2 is not None:
                # Parameter added
                diff[param] = {
                    'type': 'added',
                    'new_value': val2,
                    'old_value': None,
                    'change': val2
                }
            elif val1 is not None and val2 is None:
                # Parameter removed
                diff[param] = {
                    'type': 'removed',
                    'new_value': None,
                    'old_value': val1,
                    'change': -val1
                }
            elif val1 != val2:
                # Parameter modified
                diff[param] = {
                    'type': 'modified',
                    'new_value': val2,
                    'old_value': val1,
                    'change': val2 - val1,
                    'relative_change': (val2 - val1) / val1 if val1 != 0 else float('inf')
                }
        
        return diff
    
    @staticmethod
    def format_diff(diff: Dict[str, Dict[str, Any]], 
                   precision: int = 6) -> str:
        """Format difference for human reading"""
        if not diff:
            return "No differences found."
        
        lines = []
        lines.append(f"Parameter differences ({len(diff)} changes):")
        lines.append("=" * 50)
        
        for param, change_info in diff.items():
            change_type = change_info['type']
            
            if change_type == 'added':
                lines.append(f"+ {param}: {change_info['new_value']:.{precision}f} (added)")
            elif change_type == 'removed':
                lines.append(f"- {param}: {change_info['old_value']:.{precision}f} (removed)")
            elif change_type == 'modified':
                old_val = change_info['old_value']
                new_val = change_info['new_value']
                change = change_info['change']
                rel_change = change_info.get('relative_change', 0)
                
                sign = '+' if change >= 0 else ''
                lines.append(
                    f"~ {param}: {old_val:.{precision}f} -> {new_val:.{precision}f} "
                    f"({sign}{change:.{precision}f}, {rel_change*100:+.2f}%)"
                )
        
        return "\n".join(lines)


class VersionManager(BaseManager):
    """
    Manager for parameter version control and provenance tracking
    
    This class provides Git-like functionality for parameter management,
    including branching, merging, tagging, and history tracking.
    
    Mathematical Expression:
        Version history forms a directed acyclic graph (DAG):
        V = {v_1, v_2, ..., v} (set of versions)
        E = {(v, v) | v is child of v} (parent-child relationships)
        
    References:
        - Peng (2011). Reproducible research in computational science.
        - FAIR data principles for scientific parameter management
    """
    
    def __init__(self, storage_dir: Optional[Path] = None, **kwargs):
        """
        Initialize version manager
        
        Parameters:
            storage_dir: Directory for storing version data
        """
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)
        self.storage_dir = storage_dir
        
        # Version storage (will be initialized in _do_initialize)
        self.versions_dir: Optional[Path] = None
        
        # Branch and tag storage
        self.refs_dir: Optional[Path] = None
        self.branches_dir: Optional[Path] = None
        self.tags_dir: Optional[Path] = None
        
        # Current state
        self.current_branch = "main"
        self.head_version_id: Optional[str] = None
        
        # Use reentrant lock because create_version may call create_tag internally.
        self._lock = threading.RLock()
    
    def _create_metadata(self) -> ManagerMetadata:
        """Create metadata for VersionManager"""
        from ...core.managers.metadata import (
            ManagerCapability, CapabilityType, 
            ManagerResource, ResourceType
        )
        
        capabilities = [
            ManagerCapability(type=CapabilityType.VERSION_CONTROL),
            ManagerCapability(type=CapabilityType.FILE_IO),
            ManagerCapability(type=CapabilityType.LIFECYCLE_MANAGEMENT)
        ]
        
        resources = [
            ManagerResource(
                type=ResourceType.PERSISTENT_STORAGE,
                mode="readwrite",
                quantity=None,
                unit="MB"
            ),
            ManagerResource(
                type=ResourceType.SYSTEM_MEMORY,
                mode="readwrite", 
                quantity=None,
                unit="MB"
            )
        ]
        
        return ManagerMetadata(
            name="VersionManager", 
            version="2.0.0",
            description="Parameter version control and provenance tracking manager",
            capabilities=capabilities,
            dependencies=[],
            resources=resources,
            custom_attributes={
                "supports_git_like_operations": True,
                "supports_branching": True,
                "supports_tagging": True,
                "max_versions_recommended": 1000
            }
        )
    
    def _do_initialize(self) -> None:
        """Initialize version manager storage.

        The version history is persistent provenance, so its directory is always
        declared (constructor or configure()); it is never a temporary directory.
        """
        if self.storage_dir is None:
            raise ValueError(
                "VersionManager requires storage_dir (constructor argument or "
                "configure({'storage_dir': ...}) before initialize())"
            )

        self.storage_dir = Path(self.storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Version storage
        self.versions_dir = self.storage_dir / "versions"
        self.versions_dir.mkdir(exist_ok=True)
        
        # Branch and tag storage
        self.refs_dir = self.storage_dir / "refs"
        self.branches_dir = self.refs_dir / "branches"
        self.tags_dir = self.refs_dir / "tags"
        self.refs_dir.mkdir(exist_ok=True)
        self.branches_dir.mkdir(exist_ok=True)
        self.tags_dir.mkdir(exist_ok=True)
        
        # Load or initialize state
        self._load_state()
    
    def _do_configure(self, config: Dict[str, Any]) -> None:
        """Configure version manager settings"""
        settings = dict(config) if config else {}
        if 'storage_dir' in settings:
            if self.state == ManagerState.UNINITIALIZED:
                self.storage_dir = Path(settings['storage_dir'])
            else:
                self.logger.warning("Cannot change storage_dir after initialization")
        
        if 'current_branch' in settings:
            self.current_branch = settings['current_branch']
    
    def _do_validate(self) -> List[str]:
        """Validate version manager state"""
        errors = []
        
        # Check storage directories exist
        if not self.storage_dir or not self.storage_dir.exists():
            errors.append("Storage directory does not exist")
        
        if not self.versions_dir or not self.versions_dir.exists():
            errors.append("Versions directory does not exist")
        
        if not self.refs_dir or not self.refs_dir.exists():
            errors.append("References directory does not exist")
        
        # Check current branch exists
        if self.current_branch:
            branch_file = self.branches_dir / self.current_branch
            if not branch_file.exists():
                errors.append(f"Current branch '{self.current_branch}' does not exist")
        
        # Check HEAD version exists if set
        if self.head_version_id:
            version_file = self.versions_dir / f"{self.head_version_id}.json"
            if not version_file.exists():
                errors.append(f"HEAD version '{self.head_version_id}' does not exist")
        
        return errors
    
    def _do_cleanup(self) -> None:
        """Clean up version manager resources"""
        # No active resources to clean up, just reset state
        self.head_version_id = None
    
    def _load_state(self):
        """Load current state from storage"""
        head_file = self.refs_dir / "HEAD"
        if head_file.exists():
            with open(head_file, 'r', encoding="utf-8-sig") as f:
                head_content = f.read().strip()
                if head_content.startswith("ref: refs/branches/"):
                    self.current_branch = head_content.split("/")[-1]
                    branch_file = self.branches_dir / self.current_branch
                    if branch_file.exists():
                        with open(branch_file, 'r', encoding="utf-8-sig") as bf:
                            self.head_version_id = bf.read().strip()
                else:
                    # Detached HEAD
                    self.head_version_id = head_content
        else:
            # Initialize main branch
            self._save_state()
    
    def _save_state(self):
        """Save current state to storage"""
        head_file = self.refs_dir / "HEAD"
        with open(head_file, 'w', encoding="utf-8") as f:
            f.write(f"ref: refs/branches/{self.current_branch}")
        
        branch_file = self.branches_dir / self.current_branch
        with open(branch_file, 'w', encoding="utf-8") as f:
            f.write(self.head_version_id or "")
    
    def create_version(self,
                      parameters: Dict[str, float],
                      message: Optional[str] = None,
                      author: Optional[str] = None,
                      calibration_result: Optional[CalibrationResult] = None,
                      tags: Optional[List[str]] = None) -> str:
        """
        Create a new parameter version
        
        Parameters:
            parameters: Parameter values
            message: Commit message
            author: Author name
            calibration_result: Associated calibration result
            tags: Tags to apply to this version
            
        Returns:
            Version ID
        """
        with self._lock:
            # Generate version ID
            version_id = str(uuid.uuid4())
            
            # Compute changes from previous version
            changes = []
            if self.head_version_id:
                previous_version = self.get_version(self.head_version_id)
                if previous_version:
                    diff = ParameterDiff.compute_diff(
                        previous_version,
                        ParameterVersion(
                            parameters,
                            VersionMetadata(version_id=version_id, branch_name=self.current_branch),
                        ),
                    )
                    for param, change_info in diff.items():
                        change_type = {
                            'added': ParameterChangeType.CREATED,
                            'removed': ParameterChangeType.DELETED,
                            'modified': ParameterChangeType.MODIFIED
                        }[change_info['type']]
                        
                        changes.append(ParameterChange(
                            parameter_name=param,
                            change_type=change_type,
                            old_value=change_info.get('old_value'),
                            new_value=change_info.get('new_value'),
                            reason=message
                        ))
            else:
                # First version - all parameters are created
                for param, value in parameters.items():
                    changes.append(ParameterChange(
                        parameter_name=param,
                        change_type=ParameterChangeType.CREATED,
                        new_value=value,
                        reason=message
                    ))
            
            # Create version metadata
            metadata = VersionMetadata(
                version_id=version_id,
                parent_version_id=self.head_version_id,
                branch_name=self.current_branch,
                author=author,
                message=message,
                tags=tags or [],
                calibration_result=calibration_result,
                changes=changes
            )
            
            # Create version
            version = ParameterVersion(parameters=parameters, metadata=metadata)
            version.metadata.checksum = version.compute_checksum()
            
            # Save version
            version_file = self.versions_dir / f"{version_id}.json"
            with open(version_file, 'w', encoding="utf-8") as f:
                json.dump(version.to_dict(), f, indent=2)
            
            # Update HEAD
            self.head_version_id = version_id
            self._save_state()
            
            # Apply tags
            for tag in tags or []:
                self.create_tag(tag, version_id)
            
            self.logger.info(f"Created version {version_id[:8]} on branch {self.current_branch}")
            return version_id
    
    def get_version(self, version_id: str) -> Optional[ParameterVersion]:
        """Get parameter version by ID; None when no such version exists.

        A version file that cannot be read or parsed raises.
        """
        version_file = self.versions_dir / f"{version_id}.json"
        if not version_file.exists():
            return None

        with open(version_file, 'r', encoding="utf-8-sig") as f:
            data = json.load(f)
        return ParameterVersion.from_dict(data)
    
    def get_current_version(self) -> Optional[ParameterVersion]:
        """Get current HEAD version"""
        if self.head_version_id:
            return self.get_version(self.head_version_id)
        return None
    
    def list_versions(self, 
                     branch: Optional[str] = None,
                     limit: Optional[int] = None) -> List[ParameterVersion]:
        """
        List versions in chronological order
        
        Parameters:
            branch: Branch to list (None = current branch)
            limit: Maximum number of versions to return
            
        Returns:
            List of parameter versions
        """
        if branch is None:
            branch = self.current_branch
        
        # Get branch head
        branch_file = self.branches_dir / branch
        if not branch_file.exists():
            return []
        
        with open(branch_file, 'r', encoding="utf-8-sig") as f:
            current_version_id = f.read().strip()
        
        # Walk back through version history
        versions = []
        while current_version_id and (limit is None or len(versions) < limit):
            version = self.get_version(current_version_id)
            if version:
                versions.append(version)
                current_version_id = version.metadata.parent_version_id
            else:
                break
        
        return versions
    
    def create_branch(self, branch_name: str, 
                     source_version_id: Optional[str] = None) -> bool:
        """
        Create a new branch
        
        Parameters:
            branch_name: Name of new branch
            source_version_id: Version to branch from (None = current HEAD)
            
        Returns:
            True if successful
        """
        with self._lock:
            branch_file = self.branches_dir / branch_name
            if branch_file.exists():
                self.logger.warning(f"Branch {branch_name} already exists")
                return False
            
            source_id = source_version_id or self.head_version_id
            if not source_id:
                self.logger.error("No version to branch from")
                return False
            
            # Create branch reference
            with open(branch_file, 'w', encoding="utf-8") as f:
                f.write(source_id)
            
            self.logger.info(f"Created branch {branch_name} from {source_id[:8]}")
            return True
    
    def checkout_branch(self, branch_name: str) -> bool:
        """
        Switch to a different branch
        
        Parameters:
            branch_name: Name of branch to switch to
            
        Returns:
            True if successful
        """
        with self._lock:
            branch_file = self.branches_dir / branch_name
            if not branch_file.exists():
                self.logger.error(f"Branch {branch_name} does not exist")
                return False
            
            # Switch branch
            self.current_branch = branch_name
            with open(branch_file, 'r', encoding="utf-8-sig") as f:
                self.head_version_id = f.read().strip()
            
            self._save_state()
            self.logger.info(f"Switched to branch {branch_name}")
            return True
    
    def list_branches(self) -> List[str]:
        """List all branches"""
        branches = []
        for branch_file in self.branches_dir.glob("*"):
            if branch_file.is_file():
                branches.append(branch_file.name)
        return sorted(branches)
    
    def create_tag(self, tag_name: str, version_id: Optional[str] = None) -> bool:
        """
        Create a tag pointing to a version
        
        Parameters:
            tag_name: Name of tag
            version_id: Version to tag (None = current HEAD)
            
        Returns:
            True if successful
        """
        with self._lock:
            tag_file = self.tags_dir / tag_name
            if tag_file.exists():
                self.logger.warning(f"Tag {tag_name} already exists")
                return False
            
            target_id = version_id or self.head_version_id
            if not target_id:
                self.logger.error("No version to tag")
                return False
            
            # Create tag reference
            with open(tag_file, 'w', encoding="utf-8") as f:
                f.write(target_id)
            
            self.logger.info(f"Created tag {tag_name} pointing to {target_id[:8]}")
            return True
    
    def get_tag(self, tag_name: str) -> Optional[str]:
        """Get version ID for a tag"""
        tag_file = self.tags_dir / tag_name
        if tag_file.exists():
            with open(tag_file, 'r', encoding="utf-8-sig") as f:
                return f.read().strip()
        return None
    
    def list_tags(self) -> List[str]:
        """List all tags"""
        tags = []
        for tag_file in self.tags_dir.glob("*"):
            if tag_file.is_file():
                tags.append(tag_file.name)
        return sorted(tags)
    
    def rollback_to_version(self, version_id: str, 
                           create_branch: bool = False,
                           branch_name: Optional[str] = None) -> bool:
        """
        Rollback to a previous version
        
        Parameters:
            version_id: Version to rollback to
            create_branch: Whether to create a new branch for rollback
            branch_name: Name of rollback branch (auto-generated if None)
            
        Returns:
            True if successful
        """
        with self._lock:
            # Verify version exists
            version = self.get_version(version_id)
            if not version:
                self.logger.error(f"Version {version_id} not found")
                return False
            
            if create_branch:
                # Create rollback branch
                if branch_name is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    branch_name = f"rollback_{version_id[:8]}_{timestamp}"
                
                if not self.create_branch(branch_name, self.head_version_id):
                    return False
                
                if not self.checkout_branch(branch_name):
                    return False
            
            # Create new version with rolled-back parameters
            rollback_message = f"Rollback to version {version_id[:8]}"
            new_version_id = self.create_version(
                parameters=version.parameters,
                message=rollback_message,
                author="system"
            )
            
            self.logger.info(f"Rolled back to {version_id[:8]} as {new_version_id[:8]}")
            return True
    
    def diff_versions(self, 
                     version1_id: str, 
                     version2_id: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Compare two versions
        
        Parameters:
            version1_id: First version ID
            version2_id: Second version ID
            
        Returns:
            Difference dictionary or None if versions not found
        """
        version1 = self.get_version(version1_id)
        version2 = self.get_version(version2_id)
        
        if not version1 or not version2:
            return None
        
        return ParameterDiff.compute_diff(version1, version2)
    
    def get_parameter_history(self, 
                             parameter_name: str,
                             limit: Optional[int] = None) -> List[Tuple[str, float, datetime]]:
        """
        Get history of a specific parameter
        
        Parameters:
            parameter_name: Name of parameter to track
            limit: Maximum number of history entries
            
        Returns:
            List of (version_id, value, timestamp) tuples
        """
        history = []
        versions = self.list_versions(limit=limit)
        
        for version in versions:
            if parameter_name in version.parameters:
                history.append((
                    version.metadata.version_id,
                    version.parameters[parameter_name],
                    version.metadata.timestamp
                ))
        
        return history
    
    def export_version(self, 
                      version_id: str,
                      export_path: Path,
                      format: str = 'json') -> None:
        """
        Export a version to file

        Parameters:
            version_id: Version to export
            export_path: Export file path
            format: Export format ('json', 'yaml', 'csv')

        Raises:
            KeyError: If the version does not exist.
            ValueError: If the format is not supported. I/O errors propagate.
        """
        if format not in ('json', 'yaml', 'csv'):
            raise ValueError(f"Unsupported export format: {format} (expected json, yaml or csv)")
        version = self.get_version(version_id)
        if version is None:
            raise KeyError(f"No parameter version '{version_id}' to export")

        export_path.parent.mkdir(parents=True, exist_ok=True)
        if format == 'json':
            with open(export_path, 'w', encoding="utf-8") as f:
                json.dump(version.to_dict(), f, indent=2)
        elif format == 'yaml':
            import yaml
            with open(export_path, 'w', encoding="utf-8") as f:
                yaml.dump(version.to_dict(), f, default_flow_style=False)
        else:
            import csv
            with open(export_path, 'w', newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(['parameter', 'value'])
                for param, value in version.parameters.items():
                    writer.writerow([param, value])
    
    def import_version(self,
                      import_path: Path,
                      message: Optional[str] = None,
                      author: Optional[str] = None) -> str:
        """
        Import a version from file
        
        Parameters:
            import_path: Import file path
            message: Import message
            author: Author name

        Returns:
            Version ID of the imported version

        Raises:
            ValueError: If the file format is not supported or the content is
                not a parameter mapping. Read and parse errors propagate.
        """
        if import_path.suffix not in ('.json', '.yaml', '.yml'):
            raise ValueError(
                f"Unsupported import format: {import_path.suffix} (expected .json, .yaml or .yml)"
            )
        with open(import_path, 'r', encoding="utf-8-sig") as f:
            if import_path.suffix == '.json':
                data = json.load(f)
            else:
                import yaml
                data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{import_path} does not hold a parameter mapping")

        # A version export holds its values under 'parameters'; otherwise the
        # file is a flat parameter mapping.
        parameters = data['parameters'] if 'parameters' in data else data

        import_message = message or f"Imported from {import_path.name}"
        return self.create_version(
            parameters=parameters,
            message=import_message,
            author=author
        )
    
    def cleanup_old_versions(self, 
                           keep_days: int = 30,
                           keep_tagged: bool = True,
                           dry_run: bool = True) -> List[str]:
        """
        Clean up old versions
        
        Parameters:
            keep_days: Keep versions newer than this many days
            keep_tagged: Keep tagged versions
            dry_run: Only report what would be deleted
            
        Returns:
            List of version IDs that were/would be deleted
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=keep_days)
        
        # Get all versions
        all_versions = []
        for version_file in self.versions_dir.glob("*.json"):
            version_id = version_file.stem
            version = self.get_version(version_id)
            if version:
                all_versions.append(version)
        
        # Get tagged versions
        tagged_versions = set()
        if keep_tagged:
            for tag_file in self.tags_dir.glob("*"):
                with open(tag_file, 'r', encoding="utf-8-sig") as f:
                    tagged_versions.add(f.read().strip())
        
        # Find versions to delete
        to_delete = []
        for version in all_versions:
            version_id = version.metadata.version_id
            
            # Skip if tagged and we're keeping tagged versions
            if keep_tagged and version_id in tagged_versions:
                continue
            
            # Skip if newer than cutoff
            if version.metadata.timestamp and version.metadata.timestamp > cutoff_date:
                continue
            
            # Skip if it's a branch head
            is_branch_head = False
            for branch_file in self.branches_dir.glob("*"):
                with open(branch_file, 'r', encoding="utf-8-sig") as f:
                    if f.read().strip() == version_id:
                        is_branch_head = True
                        break
            
            if is_branch_head:
                continue
            
            to_delete.append(version_id)
        
        # Delete versions (if not dry run)
        if not dry_run:
            for version_id in to_delete:
                version_file = self.versions_dir / f"{version_id}.json"
                if version_file.exists():
                    version_file.unlink()
                    self.logger.info(f"Deleted old version {version_id[:8]}")
        
        return to_delete
    
    @contextmanager
    def transaction(self):
        """Context manager for transactional operations"""
        with self._lock:
            # Save current state
            original_head = self.head_version_id
            original_branch = self.current_branch
            
            try:
                yield
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError):
                # Rollback on error
                self.head_version_id = original_head
                self.current_branch = original_branch
                self._save_state()
                raise


@dataclass
class VersionedParameter:
    """A parameter with version control capabilities"""
    name: str
    current_value: float
    version_manager: Optional[VersionManager] = None
    units: Optional[str] = None
    description: Optional[str] = None
    bounds: Optional[Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.version_manager is None:
            raise ValueError("version_manager must be provided explicitly")
    
    def update_value(self, 
                    new_value: float,
                    message: Optional[str] = None,
                    author: Optional[str] = None) -> str:
        """Update parameter value and create version"""
        if self.bounds:
            if not (self.bounds[0] <= new_value <= self.bounds[1]):
                raise ValueError(f"Value {new_value} outside bounds {self.bounds}")
        
        old_value = self.current_value
        self.current_value = new_value
        
        # Create version
        return self.version_manager.create_version(
            parameters={self.name: new_value},
            message=message or f"Updated {self.name} from {old_value} to {new_value}",
            author=author
        )
    
    def get_history(self, limit: Optional[int] = None) -> List[Tuple[str, float, datetime]]:
        """Get parameter history"""
        return self.version_manager.get_parameter_history(self.name, limit)
    
    def rollback_to_version(self, version_id: str) -> bool:
        """Rollback to specific version"""
        version = self.version_manager.get_version(version_id)
        if version and self.name in version.parameters:
            old_value = self.current_value
            self.current_value = version.parameters[self.name]
            
            # Create rollback version
            self.version_manager.create_version(
                parameters={self.name: self.current_value},
                message=f"Rollback {self.name} from {old_value} to {self.current_value}",
                author="system"
            )
            return True
        return False
