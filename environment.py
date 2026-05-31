"""Dependency notes for the OEOC orbit-configuration model."""

ENVIRONMENT_NAME = "EOC-configuration"

REQUIRED_PACKAGES = {
    "python": ">=3.11",
    "numpy": ">=1.24",
    "scipy": ">=1.10",
    "matplotlib": ">=3.7",
    "pillow": ">=9.5",
}


def print_environment():
    """Prints the suggested local environment name and package list."""
    print(ENVIRONMENT_NAME)
    for package, version in REQUIRED_PACKAGES.items():
        print(f"{package}{version}")


if __name__ == "__main__":
    print_environment()
