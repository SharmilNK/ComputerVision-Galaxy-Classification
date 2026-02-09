# How to Push to Your Own GitHub Repository

## Step 1: Create a New Repository on GitHub
1. Go to https://github.com/new
2. Repository name: `galaxy-morphology-ui` (or any name you like)
3. Make it **Public** (so Vercel can access it)
4. **Don't** initialize with README, .gitignore, or license
5. Click "Create repository"

## Step 2: Connect Your Local Code
Run these commands in your terminal (from the Tiff directory):

```bash
# Remove the old remote
git remote remove origin

# Add your new repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/galaxy-morphology-ui.git

# Push your code
git push -u origin main
```

## Step 3: Import to Vercel
1. Go back to Vercel
2. Click "Import Git Repository"
3. Install GitHub app if needed
4. Select your new repository
5. Configure:
   - Root Directory: `galaxy-ui`
   - Build Command: `npm run build`
   - Output Directory: `dist`
6. Click "Deploy"
