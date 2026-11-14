"""
Script to fix conservation check type errors in all biosphere modules.
"""
import re
from pathlib import Path

# List of files to fix
files_to_fix = [
    r"relic\scientific\biosphere\respiration\maintenance_respiration.py",
    r"relic\scientific\biosphere\respiration\growth_respiration.py",
    r"relic\scientific\biosphere\growth\biomass_allocation.py",
    r"relic\scientific\biosphere\growth\structural_growth.py",
    r"relic\scientific\biosphere\phenology\base.py",
    r"relic\scientific\biosphere\stress\water_stress.py",
    r"relic\scientific\biosphere\stress\nutrient_stress.py",
    r"relic\scientific\biosphere\stress\thermal_stress.py",
    r"relic\scientific\biosphere\stress\ozone_stress.py",
    r"relic\scientific\biosphere\bvoc\megan_model.py",
    r"relic\scientific\biosphere\bvoc\synthesis\isoprene.py",
    r"relic\scientific\biosphere\bvoc\synthesis\monoterpene.py",
    r"relic\scientific\biosphere\bvoc\synthesis\sesquiterpene.py",
]

def fix_conservation_check(content: str) -> tuple[str, bool]:
    """
    Fix conservation check type errors in file content.

    Returns:
        tuple: (fixed_content, was_modified)
    """
    # Pattern to match the conservation check code
    pattern = r'(\s+# Conservation check\s+if self\.metadata\.conservation_laws:)\s+# Get conservation tolerance from parameters\s+conservation_tolerance = self\.get_parameter_value\((.*?)\)\s+(.*?)strict_conservation = self\.get_parameter_value\((.*?)\)\s+# Check conservation laws\s+violations = self\.check_conservation_laws\(state, output\)\s+for law, violation in violations\.items\(\):\s+if abs\(violation\) > conservation_tolerance:'

    # Replacement pattern with proper type conversion
    replacement = r'''\1
            # Get conservation tolerance from parameters
            conservation_tolerance_param = self.get_parameter_value(\2)
            # Convert to Python float for comparison
            conservation_tolerance = float(conservation_tolerance_param.item()) if isinstance(conservation_tolerance_param, torch.Tensor) else float(conservation_tolerance_param)

            \3strict_conservation_param = self.get_parameter_value(\4)
            # Convert to Python bool for comparison
            strict_conservation = bool(strict_conservation_param.item()) if isinstance(strict_conservation_param, torch.Tensor) else bool(strict_conservation_param)

            # Check conservation laws
            violations = self.check_conservation_laws(state, output)

            for law, violation in violations.items():
                if abs(violation) > conservation_tolerance:'''

    # Check if pattern exists
    if re.search(pattern, content, re.DOTALL):
        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        return new_content, True

    return content, False

def main():
    """Main function to fix all files."""
    base_dir = Path(__file__).resolve().parents[1]

    total_files = 0
    fixed_files = 0
    skipped_files = 0

    for file_path in files_to_fix:
        full_path = base_dir / file_path

        if not full_path.exists():
            print(f"[SKIP] File not found: {file_path}")
            skipped_files += 1
            continue

        try:
            # Read file content
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Fix conservation checks
            fixed_content, was_modified = fix_conservation_check(content)

            if was_modified:
                # Write back
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                print(f"[OK] Fixed: {file_path}")
                fixed_files += 1
            else:
                print(f"[SKIP] No conservation check found: {file_path}")
                skipped_files += 1

            total_files += 1

        except Exception as e:
            print(f"[ERROR] Failed to process {file_path}: {e}")

    print(f"\n=== Summary ===")
    print(f"Total files processed: {total_files}")
    print(f"Files fixed: {fixed_files}")
    print(f"Files skipped: {skipped_files}")

if __name__ == "__main__":
    main()

