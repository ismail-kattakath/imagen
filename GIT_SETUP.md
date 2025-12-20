# Git Setup Guide

Repository has been initialized with initial commit. Follow these steps to complete setup.

## Current Status

- ✅ Git repository initialized
- ✅ Initial commit created (fb72853)
- ✅ 69 files committed (4,595 lines)
- ✅ .env excluded from version control
- ✅ Models directory excluded (large files)
- ⏳ Remote repository not yet configured

## Quick Setup

### 1. Create Remote Repository

Create a new repository on your preferred platform:

- **GitHub:** https://github.com/new
- **GitLab:** https://gitlab.com/projects/new
- **Bitbucket:** https://bitbucket.org/repo/create

**Recommended settings:**
- Name: `imagen` or `imagen-ai`
- Visibility: Private (contains GCP configuration examples)
- Don't initialize with README (we already have one)

### 2. Connect and Push

```bash
# Add remote (replace with your URL)
git remote add origin https://github.com/YOUR_USERNAME/imagen.git

# Push to remote
git push -u origin main

# Verify
git remote -v
```

### 3. Optional: Create Development Branch

```bash
# Create and switch to develop branch
git checkout -b develop

# Push develop branch
git push -u origin develop

# Switch back to main
git checkout main
```

### 4. Optional: Add Version Tag

```bash
# Create annotated tag for v1.0.0
git tag -a v1.0.0 -m "Initial release - Complete platform"

# Push tags
git push origin v1.0.0

# Or push all tags
git push --tags
```

## Recommended Workflow

### Branch Strategy

```
main
 └── develop
      ├── feature/add-new-pipeline
      ├── feature/improve-caching
      └── hotfix/fix-background-removal
```

**Branches:**
- `main` - Production-ready code only
- `develop` - Integration branch for features
- `feature/*` - New features and enhancements
- `hotfix/*` - Critical bug fixes
- `release/*` - Release preparation

### Creating Feature Branch

```bash
# From develop branch
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat(scope): description"

# Push feature branch
git push -u origin feature/your-feature-name

# Create pull request on GitHub/GitLab
```

### Commit Message Convention

Follow conventional commits format:

```
type(scope): subject

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

**Examples:**
```bash
git commit -m "feat(api): add rate limiting middleware"
git commit -m "fix(pipeline): correct background removal alpha channel"
git commit -m "docs(deployment): update GKE deployment steps"
git commit -m "chore(deps): update diffusers to v0.26.0"
```

## GitHub/GitLab Configuration

### Protected Branches

Protect `main` and `develop` branches:

**GitHub:**
1. Settings → Branches → Add rule
2. Branch name pattern: `main`
3. Enable:
   - Require pull request before merging
   - Require approvals (1+)
   - Require status checks to pass
   - Include administrators

**GitLab:**
1. Settings → Repository → Protected branches
2. Select `main` branch
3. Allowed to merge: Maintainers
4. Allowed to push: No one

### Branch Protection Rules

```yaml
# .github/workflows/branch-protection.yml
name: Branch Protection

on:
  pull_request:
    branches: [main, develop]

jobs:
  tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: make test
      
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Lint code
        run: make lint
```

## Useful Git Commands

### Status and History

```bash
# Check status
git status

# View commit history
git log --oneline --graph --all

# View changes
git diff

# View file history
git log --follow -- path/to/file
```

### Syncing

```bash
# Pull latest changes
git pull origin main

# Fetch all branches
git fetch --all

# Update all branches
git pull --all
```

### Cleanup

```bash
# Delete local merged branches
git branch --merged | grep -v "\*" | xargs -n 1 git branch -d

# Prune deleted remote branches
git remote prune origin

# Clean untracked files (dry run first!)
git clean -n
git clean -fd
```

## Common Workflows

### Update from Main

```bash
# On feature branch
git checkout feature/your-feature
git fetch origin
git rebase origin/main

# Or merge
git merge origin/main
```

### Squash Commits Before PR

```bash
# Squash last 3 commits
git rebase -i HEAD~3

# Mark commits as 'squash' or 's'
# Save and edit commit message
```

### Cherry-pick Commit

```bash
# Apply specific commit to current branch
git cherry-pick <commit-hash>
```

### Undo Changes

```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Revert commit (create new commit)
git revert <commit-hash>
```

## CI/CD Integration

### GitHub Actions Example

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -e ".[dev]"
      
      - name: Run tests
        run: make test
      
      - name: Lint
        run: make lint
```

### GitLab CI Example

Create `.gitlab-ci.yml`:

```yaml
stages:
  - test
  - lint

test:
  stage: test
  image: python:3.11
  script:
    - pip install -e ".[dev]"
    - make test

lint:
  stage: lint
  image: python:3.11
  script:
    - pip install -e ".[dev]"
    - make lint
```

## Troubleshooting

### Large Files Error

If you accidentally committed large model files:

```bash
# Remove from history
git filter-branch --tree-filter 'rm -rf models/*' HEAD

# Or use git-filter-repo (recommended)
git filter-repo --path models/ --invert-paths
```

### Rejected Push

```bash
# Pull with rebase
git pull --rebase origin main

# Resolve conflicts if any
git rebase --continue

# Push
git push origin main
```

### Wrong Commit Message

```bash
# Amend last commit message
git commit --amend -m "New message"

# Force push (only if not pushed yet!)
git push --force-with-lease
```

## Best Practices

1. **Commit Often:** Small, focused commits are better
2. **Write Clear Messages:** Follow conventional commits
3. **Pull Before Push:** Always sync before pushing
4. **Review Before Commit:** Use `git diff` to review changes
5. **Use .gitignore:** Never commit secrets or large files
6. **Tag Releases:** Use semantic versioning (v1.0.0)
7. **Protect Main:** Require pull requests for main
8. **Test Before Merge:** Run tests before merging
9. **Document Changes:** Update CHANGELOG.md
10. **Sign Commits:** Use GPG signing for security

## Next Steps

1. Create remote repository
2. Push initial commit
3. Set up branch protection
4. Configure CI/CD (optional)
5. Add collaborators (if team project)
6. Create first feature branch
7. Start development!

---

**Note:** The `.env` file is excluded from git for security. Each developer should create their own `.env` from `.env.example`.
