#!/bin/bash

# Python Package Template Setup Script
# This script initializes a new Python package from this template

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to convert repository name to package name (PyPI format)
repo_to_package_name() {
    local repo_name="$1"
    # Convert to lowercase
    echo "$repo_name" | tr '[:upper:]' '[:lower:]'
}

# Function to convert repository name to module name (Python import)
repo_to_module_name() {
    local repo_name="$1"
    # Convert to lowercase and replace hyphens with underscores
    echo "$repo_name" | tr '[:upper:]' '[:lower:]' | sed 's/-/_/g'
}

# Function to extract package name from repo name (for ds-common-<name>-py-lib pattern)
extract_package_name() {
    local repo_name="$1"
    # Remove ds-common- prefix and -py-lib suffix if present
    echo "$repo_name" | sed 's/^ds-common-//' | sed 's/-py-lib$//'
}

# Function to get current date in YYYY-MM-DD format
get_current_date() {
    date +"%Y-%m-%d"
}

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    print_error "This script must be run in a git repository root directory"
    exit 1
fi

# Get repository name from git remote or prompt user
if command -v git >/dev/null 2>&1; then
    REPO_NAME=$(git remote get-url origin 2>/dev/null | sed 's/.*\///' | sed 's/\.git$//' || echo "")
fi

# If we couldn't get repo name from git, prompt user
if [ -z "$REPO_NAME" ]; then
    echo -n "Enter the repository name (e.g., ds-common-logging-py-lib): "
    read -r REPO_NAME
fi

if [ -z "$REPO_NAME" ]; then
    print_error "Repository name cannot be empty"
    exit 1
fi

# Convert repository name to various formats
PACKAGE_NAME=$(repo_to_package_name "$REPO_NAME")
MODULE_NAME=$(repo_to_module_name "$REPO_NAME")
PACKAGE_SUFFIX=$(extract_package_name "$REPO_NAME")

# Prompt for author name
echo -n "Enter author name (default: Grasp Labs AS): "
read -r AUTHOR_NAME
if [ -z "$AUTHOR_NAME" ]; then
    AUTHOR_NAME="Grasp Labs AS"
fi

# Prompt for author email
echo -n "Enter author email (default: hello@grasplabs.com): "
read -r AUTHOR_EMAIL
if [ -z "$AUTHOR_EMAIL" ]; then
    AUTHOR_EMAIL="hello@grasplabs.com"
fi

# Get current date
CURRENT_DATE=$(get_current_date)
CURRENT_YEAR=$(date +"%Y")

print_status "Setting up template with the following values:"
echo "  Repository name: $REPO_NAME"
echo "  Package name (PyPI): $PACKAGE_NAME"
echo "  Module name (Python): $MODULE_NAME"
echo "  Author: $AUTHOR_NAME <$AUTHOR_EMAIL>"
echo

# Confirm before proceeding
echo -n "Proceed with template setup? (y/N): "
read -r CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    print_warning "Template setup cancelled"
    exit 0
fi

print_status "Starting template setup..."

# Create backup of original files
print_status "Creating backup of original files..."
mkdir -p .template-backup
find . -maxdepth 1 -type f \( -name "*.md" -o -name "*.toml" -o -name "*.txt" -o -name "*.yaml" -o -name "*.yml" \) -exec cp {} .template-backup/ \; 2>/dev/null || true

# Function to replace in file
replace_in_file() {
    local file="$1"
    if [ -f "$file" ]; then
        sed -i.bak "s|{{PROJECT_NAME}}|$PACKAGE_NAME|g" "$file"
        sed -i.bak "s|{{PYTHON_MODULE_NAME}}|$MODULE_NAME|g" "$file"
        sed -i.bak "s|{{AUTHOR_NAME}}|$AUTHOR_NAME|g" "$file"
        sed -i.bak "s|{{AUTHOR_EMAIL}}|$AUTHOR_EMAIL|g" "$file"
        sed -i.bak "s|{{GITHUB_REPO}}|$REPO_NAME|g" "$file"
        sed -i.bak "s|{{DATE}}|$CURRENT_DATE|g" "$file"
        sed -i.bak "s|{{YEAR}}|$CURRENT_YEAR|g" "$file"
        rm -f "$file.bak"
    fi
}

# Update pyproject.toml
print_status "Updating pyproject.toml..."
replace_in_file "pyproject.toml"

# Update configuration files
print_status "Updating configuration files..."
replace_in_file ".config/ruff.toml"
replace_in_file ".config/pytest.ini"

# Update README.md
print_status "Updating README.md..."
replace_in_file "README.md"

# Update PyPI.md
print_status "Updating PyPI.md..."
replace_in_file "PyPI.md"

# Update SETUP.md
print_status "Updating SETUP.md..."
replace_in_file "SETUP.md"

# Update CONTRIBUTING.md
print_status "Updating CONTRIBUTING.md..."
replace_in_file "CONTRIBUTING.md"

# Update TEMPLATE_VARIABLES.md (remove template variables)
print_status "Updating TEMPLATE_VARIABLES.md..."
replace_in_file "TEMPLATE_VARIABLES.md"

# Update TEMPLATE_CHECKLIST.md
print_status "Updating TEMPLATE_CHECKLIST.md..."
replace_in_file "TEMPLATE_CHECKLIST.md"

# Update docs/source/conf.py
print_status "Updating docs/source/conf.py..."
replace_in_file "docs/source/conf.py"

# Update docs/source/index.rst
print_status "Updating docs/source/index.rst..."
replace_in_file "docs/source/index.rst"

# Update Makefile
print_status "Updating Makefile..."
replace_in_file "Makefile"

# Update GitHub workflows
print_status "Updating GitHub workflows..."
for file in .github/workflows/*.yaml; do
    if [ -f "$file" ]; then
        replace_in_file "$file"
    fi
done

# Update tests
print_status "Updating tests..."
for file in tests/*.py; do
    if [ -f "$file" ]; then
        replace_in_file "$file"
    fi
done

# Update source directory
print_status "Renaming source directory..."
if [ -d "src/{package_name}" ]; then
    mv "src/{package_name}" "src/$MODULE_NAME"
    print_success "Renamed src/{package_name} to src/$MODULE_NAME"
else
    print_warning "Source directory src/{package_name} not found, skipping rename"
fi

# Update __init__.py
print_status "Updating __init__.py..."
replace_in_file "src/$MODULE_NAME/__init__.py"

# Update example.py if it exists
if [ -f "src/$MODULE_NAME/example.py" ]; then
    print_status "Updating example.py..."
    replace_in_file "src/$MODULE_NAME/example.py"
fi

# Remove template-specific files
print_status "Cleaning up template files..."
rm -f setup-template.sh

# Update .gitignore to remove template backup
if ! grep -q ".template-backup/" .gitignore 2>/dev/null; then
    echo "" >> .gitignore
    echo "# Template backup" >> .gitignore
    echo ".template-backup/" >> .gitignore
fi

print_status "Installing pre-commit hooks..."
pre-commit install

print_success "Template setup completed successfully!"
echo
print_status "Next steps:"
echo "  1. Review and customize the generated files"
echo "  2. Update PyPI.md with your package's specific usage examples"
echo "  3. Add your package code to src/$MODULE_NAME/"
echo "  4. Update README.md with your specific documentation"
echo "  5. Install dependencies: uv sync --all-extras --dev"
echo "  6. Install pre-commit hooks: uv run pre-commit install"
echo "  7. Run tests: make test"
echo "  8. Commit your changes: git add . && git commit -m \"Initial commit from template\""
echo
print_status "Template backup files are available in .template-backup/ if you need to reference them"
