# Script to push your code to your own GitHub repository
# Replace YOUR_USERNAME with your actual GitHub username

Write-Host "Step 1: Removing old remote..." -ForegroundColor Yellow
git remote remove origin

Write-Host "`nStep 2: Please enter your GitHub username:" -ForegroundColor Yellow
$username = Read-Host "GitHub Username"

Write-Host "`nStep 3: Adding your new repository..." -ForegroundColor Yellow
git remote add origin "https://github.com/$username/galaxy-morphology-ui.git"

Write-Host "`nStep 4: Pushing your code..." -ForegroundColor Yellow
git push -u origin main

Write-Host "`n✅ Done! Now go to Vercel and import your repository." -ForegroundColor Green
