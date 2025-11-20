# Frontend API Configuration

## Environment Variables

The frontend uses environment variables to configure the ML backend API URL.

### Files

- **`.env`** - Production configuration (committed to git)
  - Uses Vercel deployment: `http://eye-glaze-srj2.vercel.app`
  - Used when deploying to production

- **`.env.local`** - Local development configuration (NOT committed)
  - Uses local backend: `http://localhost:5000`
  - Override for local development
  - Takes precedence over `.env`

- **`.env.example`** - Example configuration template
  - Copy this to create your own `.env.local`

### Usage

#### For Production (Vercel Backend)
The `.env` file is already configured with your Vercel ML backend:
```bash
VITE_API_URL=http://eye-glaze-srj2.vercel.app
```

Just build and deploy:
```bash
npm run build
```

#### For Local Development (Local Flask Backend)
The `.env.local` file is already configured for local development:
```bash
VITE_API_URL=http://localhost:5000
```

Make sure your Flask backend is running:
```bash
cd ../../Python_Backend
python api/index.py
```

Then start the frontend:
```bash
npm run dev
```

### How It Works

The `Dashboard.tsx` component reads the API URL from environment variables:

```typescript
const API_URL = import.meta.env.VITE_API_URL || 'http://eye-glaze-srj2.vercel.app';
```

All API calls use this URL:
```typescript
fetch(`${API_URL}/predict`, { ... })
```

### Vite Environment Variables

Vite exposes environment variables through `import.meta.env`:
- Variables must be prefixed with `VITE_` to be exposed to client-side code
- `.env.local` overrides `.env`
- Changes require a restart of the dev server

### Changing the Backend URL

1. **For development**: Edit `.env.local`
2. **For production**: Edit `.env` and commit
3. **Restart** the dev server after changes

### Testing Different Backends

```bash
# Test with Vercel backend
VITE_API_URL=http://eye-glaze-srj2.vercel.app npm run dev

# Test with local backend
VITE_API_URL=http://localhost:5000 npm run dev

# Test with custom backend
VITE_API_URL=https://your-custom-backend.com npm run dev
```

## Deployment

When deploying the frontend to Vercel or other platforms, make sure to set the environment variable:

**Vercel Dashboard:**
1. Go to Project Settings → Environment Variables
2. Add: `VITE_API_URL` = `http://eye-glaze-srj2.vercel.app`
3. Redeploy

The production build will use the configured backend URL.
