# How to Deploy Your Galaxy Morphology App to Vercel

## 🚀 Easiest Method: Create Your Own GitHub Repository

### Step 1: Create GitHub Repository (2 minutes)
1. Go to **https://github.com/new**
2. Repository name: `galaxy-morphology-ui`
3. Make it **Public**
4. **Don't** check any boxes (no README, no .gitignore)
5. Click **"Create repository"**

### Step 2: Push Your Code (3 minutes)
Open PowerShell in your project folder and run:

```powershell
# Go to your project root
cd C:\Users\som\Desktop\CourseSemII\DeepLearn\computer_vision\final\Git\StarStruck_2\Tiff

# Remove old remote
git remote remove origin

# Add your new repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/galaxy-morphology-ui.git

# Push your code
git push -u origin main
```

### Step 3: Deploy on Vercel (2 minutes)
1. Go to **https://vercel.com**
2. Sign up/login with GitHub
3. Click **"Add New" → "Project"**
4. Click **"Install"** next to GitHub (if needed)
5. Find and select your `galaxy-morphology-ui` repository
6. **Configure:**
   - **Root Directory:** `galaxy-ui` (click "Edit" and type this)
   - **Build Command:** `npm run build` (auto-detected)
   - **Output Directory:** `dist` (auto-detected)
7. Click **"Deploy"**
8. Wait 1-2 minutes
9. **Done!** You'll get a URL like `https://galaxy-morphology-ui.vercel.app`

## ✅ Your App Will:
- Be publicly accessible worldwide
- Automatically use your Hugging Face Space backend
- Have HTTPS (secure connection)
- Update automatically when you push to GitHub

## 📤 Share the URL:
Once deployed, share the Vercel URL with your friend!

---

## Alternative: Direct Upload (If GitHub is too complicated)

If you prefer not to use GitHub:
1. Go to **https://vercel.com**
2. Sign up/login
3. Look for **"Upload"** or **"Deploy"** button
4. Upload the `galaxy-ui/dist` folder
5. Deploy!
