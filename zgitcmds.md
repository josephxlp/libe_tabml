# Step-by-Step Guide for Git Workflow

### 1. Clone the repository (if not already cloned)
```bash
# Replace <repository-url> with your repository link
git clone <repository-url>
cd <repository-folder>
```

### 2. Create a new branch
```bash
# Replace <branch-name> with your desired branch name
git checkout -b <branch-name>
```

### 3. Add files or make changes
```bash
# Add new files or modify existing ones as needed
# For example:
echo "Hello, World!" > example.txt
```

### 4. Stage the changes
```bash
# Add specific files
git add <file1> <file2>

# Or add all changes
git add .
```

### 5. Commit the changes
```bash
# Write a meaningful commit message
git commit -m "<Your commit message>"
```

### 6. Push the new branch to the remote repository
```bash
# Push the branch to the remote repository
git push origin <branch-name>
```

### 7. Switch to the main branch
```bash
git checkout main
```

### 8. Pull the latest changes from the remote main branch
```bash
git pull origin main
```

### 9. Merge the new branch into the main branch
```bash
# Ensure you are on the main branch
git merge <branch-name>
```

### 10. Push the updated main branch to the remote repository
```bash
git push origin main
```

### Notes to Avoid Conflicts
- Ensure no one else is making changes to the same files in the main branch.
- Regularly pull updates from the main branch while working on the new branch (`git pull origin main`).
- Resolve any conflicts that arise during merging by following Git's conflict resolution prompts.



# Steps to Sync Main Branch After Merging a Pull Request

## 1. Switch to the `main` branch locally
```bash
git checkout main
```

## 2. Fetch the latest changes from the remote repository
This ensures you retrieve the latest updates from the remote `main` branch.
```bash
git fetch origin
```

## 3. Pull the latest changes into your local `main` branch
This updates your local `main` branch to match the remote.
```bash
git pull origin main
```

## 4. (Optional) Clean up your feature branch locally
If the feature branch is no longer needed after merging the PR:
```bash
git branch -d <branch-name>  # Deletes the branch locally
```

## 5. (Optional) Remove the remote branch
If you want to clean up the remote repository by removing the merged branch:
```bash
git push origin --delete <branch-name>  # Deletes the branch remotely
```

# Summary of Commands
```bash
git checkout main
git fetch origin
git pull origin main
git branch -d <branch-name>        # Optional: Delete local branch
git push origin --delete <branch-name>  # Optional: Delete remote branch



