# Deploying Eye-Glaze Backend to Vercel

This guide will help you deploy your Flask backend to Vercel.

## Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **Vercel CLI**: Install globally
   ```bash
   npm install -g vercel
   ```

## Files Created for Deployment

- `vercel.json` - Vercel configuration
- `api/index.py` - Serverless function entry point
- `requirements-vercel.txt` - Lightweight dependencies for Vercel
- `.vercelignore` - Files to exclude from deployment

## Deployment Steps

### Option 1: Deploy via Vercel CLI (Recommended)

1. **Login to Vercel**:
   ```bash
   vercel login
   ```

2. **Navigate to the backend directory**:
   ```bash
   cd EYE_GLAZE/Python_Backend
   ```

3. **Deploy to Vercel**:
   ```bash
   vercel
   ```
   
   Follow the prompts:
   - Set up and deploy? **Y**
   - Which scope? Select your account
   - Link to existing project? **N**
   - Project name? `eye-glaze-backend` (or your choice)
   - In which directory is your code located? `./`
   - Want to override settings? **N**

4. **Deploy to Production**:
   ```bash
   vercel --prod
   ```

### Option 2: Deploy via Vercel Dashboard

1. Go to [vercel.com/dashboard](https://vercel.com/dashboard)
2. Click **"Add New"** → **"Project"**
3. Import your GitHub repository
4. Configure:
   - **Framework Preset**: Other
   - **Root Directory**: `EYE_GLAZE/Python_Backend`
   - **Build Command**: (leave empty)
   - **Output Directory**: (leave empty)
5. Click **"Deploy"**

## Configuration Notes

### Why Lightweight Dependencies?

The `requirements-vercel.txt` file uses lightweight versions:
- `opencv-python-headless` instead of `opencv-python` (no GUI dependencies)
- Removed `tensorflow` and `scikit-learn` (not needed for basic detection)
- This keeps the deployment size under Vercel's limits

### Vercel Limitations

- **Function Size**: Max 50MB
- **Function Duration**: 10 seconds (Hobby), 60 seconds (Pro)
- **Memory**: 1024MB (Hobby), 3008MB (Pro)
- **No Persistent Storage**: Cannot save files between requests

### Environment Variables (Optional)

If you need to add environment variables:

1. Go to your project settings in Vercel
2. Navigate to **Environment Variables**
3. Add variables like:
   - `FLASK_ENV=production`
   - `MAX_CONTENT_LENGTH=10485760` (10MB)

## Testing Your Deployment

Once deployed, you'll get a URL like: `https://eye-glaze-backend.vercel.app`

### Test the health endpoint:
```bash
curl https://your-deployment-url.vercel.app/health
```

### Test the predict endpoint:
```bash
curl -X POST https://your-deployment-url.vercel.app/predict \
  -F "image=@test_image.jpg" \
  -F "age=30"
```

## Updating Your Frontend

After deployment, update your React frontend to use the Vercel URL:

In your React app's API configuration:
```typescript
const API_URL = 'https://your-deployment-url.vercel.app';
```

## Troubleshooting

### Build Fails

- Check deployment logs in Vercel dashboard
- Ensure `requirements-vercel.txt` has compatible versions
- Verify `api/index.py` has no syntax errors

### CORS Issues

- The `api/index.py` is configured to allow all origins
- If needed, update the CORS configuration to specific domains

### Function Timeout

- Optimize image processing
- Consider upgrading to Vercel Pro for longer timeouts
- Add timeout handling in frontend

### Large File Upload

- Vercel has a 4.5MB limit for body size
- Compress images on frontend before uploading
- Consider using a CDN for large files

## Advanced Configuration

### Custom Domain

1. Go to project settings → Domains
2. Add your custom domain
3. Configure DNS records as instructed

### Analytics

Enable Vercel Analytics in your project settings for usage monitoring.

### Logs

View real-time logs:
```bash
vercel logs <deployment-url>
```

## Cost Considerations

- **Hobby Plan**: Free, 100GB bandwidth, 100 hours function execution
- **Pro Plan**: $20/month, 1TB bandwidth, 1000 hours function execution

## Support

For issues:
- Check [Vercel Documentation](https://vercel.com/docs)
- Review deployment logs
- Test locally first with `python api/index.py`

## Local Testing

Test the Vercel function locally:

```bash
cd EYE_GLAZE/Python_Backend
python api/index.py
```

Then visit `http://localhost:5000` to test endpoints.
