"""Verification script to check app.py imports without errors."""
import sys
import traceback

def verify_imports():
    """
    Attempt to import app.py and report any errors.

    Returns
    -------
    bool
        True if imports successful, False otherwise.
    """
    try:
        # Import main app module
        import app

        # Import UI modules
        from ui import sidebar
        from ui import tab_module1
        from ui import tab_module2

        # Import module functions
        from modules.module1_protected_areas import coverage_stats
        from modules.module1_protected_areas import representativity
        from modules.module1_protected_areas import gap_analysis

        from modules.module2_favourability import export

        print("✓ All imports successful!")
        print("\nImported modules:")
        print("  - app")
        print("  - ui.sidebar")
        print("  - ui.tab_module1")
        print("  - ui.tab_module2")
        print("  - modules.module1_protected_areas.coverage_stats")
        print("  - modules.module1_protected_areas.representativity")
        print("  - modules.module1_protected_areas.gap_analysis")
        print("  - modules.module2_favourability.export")

        return True

    except Exception as e:
        print(f"✗ Import error detected!")
        print(f"\nError type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_imports()
    sys.exit(0 if success else 1)
