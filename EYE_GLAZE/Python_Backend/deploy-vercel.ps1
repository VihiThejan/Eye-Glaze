# Deploy Eye-Glaze Backend to Vercel
# This script commits changes and helps with deployment

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Eye-Glaze Backend - Vercel Deployment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to backend directory
$backendPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $backendPath
Write-Host "📁 Backend directory: $backendPath" -ForegroundColor Green

# Go to repository root
Set-Location "../.."
$repoRoot = Get-Location
Write-Host "📁 Repository root: $repoRoot" -ForegroundColor Green
Write-Host ""

# Check git status
Write-Host "📊 Checking git status..." -ForegroundColor Yellow
git status --short

Write-Host ""
Write-Host "🔄 Adding Vercel deployment files..." -ForegroundColor Yellow
git add EYE_GLAZE/Python_Backend/vercel.json
git add EYE_GLAZE/Python_Backend/api/
git add EYE_GLAZE/Python_Backend/.vercelignore
git add EYE_GLAZE/Python_Backend/requirements.txt
git add EYE_GLAZE/Python_Backend/requirements-full.txt
git add EYE_GLAZE/Python_Backend/requirements-vercel.txt
git add EYE_GLAZE/Python_Backend/README-VERCEL.md

Write-Host ""
Write-Host "💾 Committing changes..." -ForegroundColor Yellow
git commit -m "Configure backend for Vercel deployment

- Add vercel.json configuration
- Create api/index.py serverless function
- Use lightweight requirements.txt (opencv-headless)
- Add deployment documentation
- Backup full requirements as requirements-full.txt"

Write-Host ""
Write-Host "⬆️  Pushing to GitHub..." -ForegroundColor Yellow
git push origin main

Write-Host ""
Write-Host "✅ Changes pushed to GitHub!" -ForegroundColor Green
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Option 1: Deploy via Vercel Dashboard" -ForegroundColor White
Write-Host "  1. Go to https://vercel.com/dashboard" -ForegroundColor Gray
Write-Host "  2. Click 'Add New' → 'Project'" -ForegroundColor Gray
Write-Host "  3. Import your GitHub repository" -ForegroundColor Gray
Write-Host "  4. Set Root Directory: EYE_GLAZE/Python_Backend" -ForegroundColor Gray
Write-Host "  5. Click Deploy" -ForegroundColor Gray
Write-Host ""
Write-Host "Option 2: Deploy via Vercel CLI" -ForegroundColor White
Write-Host "  1. Install CLI: npm install -g vercel" -ForegroundColor Gray
Write-Host "  2. Login: vercel login" -ForegroundColor Gray
Write-Host "  3. Deploy: vercel --prod" -ForegroundColor Gray
Write-Host ""
Write-Host "📖 Full guide: EYE_GLAZE/Python_Backend/README-VERCEL.md" -ForegroundColor Yellow
Write-Host ""
