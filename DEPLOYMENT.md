# Deploying Loan Prediction App to Render

This guide will walk you through deploying your loan prediction web app to Render.

## Prerequisites

1. A GitHub account
2. Git installed on your computer
3. Your trained `loan_model.pkl` file

## Step-by-Step Deployment Guide

### Step 1: Prepare Your Project

1. **Add your model file**
   - Copy `loan_model.pkl` to the `loan_app` folder
   - Make sure it's in the same directory as `app.py`

2. **Verify files**
   - Ensure you have all these files:

     ```text
     loan_app/
     ├── app.py
     ├── templates/
     │   └── index.html
     ├── loan_model.pkl
     ├── requirements.txt
     ├── Procfile
     ├── runtime.txt
     ├── .gitignore
     └── DEPLOYMENT.md
     ```

### Step 2: Push to GitHub

1. **Initialize Git repository** (in the loan_app folder):

   ```bash
   git init
   git add .
   git commit -m "Initial commit - Loan Prediction App"
   ```

2. **Create a new repository on GitHub**:

   - Go to <https://github.com/new>
   - Name it: `loan-prediction-app` (or any name you like)
   - Don't initialize with README (we already have files)
   - Click "Create repository"

3. **Push your code**:

   ```bash
   git remote add origin <https://github.com/YOUR-USERNAME/loan-prediction-app.git>
   git branch -M main
   git push -u origin main
   ```

### Step 3: Deploy on Render

1. **Sign up/Login to Render**:
   - Go to <https://render.com>
   - Sign up for free account (or login)
   - You can sign up with your GitHub account

2. **Create a New Web Service**:
   - Click "New +" button in the dashboard
   - Select "Web Service"
   - Connect your GitHub account if not already connected
   - Select your `loan-prediction-app` repository

3. **Configure the Web Service**:
   Fill in these settings:

   - **Name**: `loan-prediction-app` (or any unique name)
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Root Directory**: Leave blank (or `.` if needed)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: `Free`

4. **Environment Variables** (if needed):
   - Usually not needed for this simple app
   - Click "Advanced" if you want to add any

5. **Click "Create Web Service"**

### Step 4: Wait for Deployment

- Render will start building your app
- This usually takes 2-5 minutes
- You'll see the build logs in real-time
- Once it says "Your service is live 🎉", you're done!

### Step 5: Access Your App

- Your app URL will be: `https://your-app-name.onrender.com`
- Click on the URL to open your loan prediction app
- Share this URL with anyone!

## Important Notes

### Free Tier Limitations

- The free tier spins down after 15 minutes of inactivity
- First request after inactivity might take 30-60 seconds to wake up
- This is normal for free tier - subsequent requests are fast
- For always-on service, upgrade to paid tier ($7/month)

### Model File Size

- Make sure `loan_model.pkl` is included in your Git repository
- If it's too large (>100MB), you might need to use Git LFS:

  ```bash
  git lfs install
  git lfs track "*.pkl"
  git add .gitattributes
  git commit -m "Track model file with LFS"
  ```

### Troubleshooting

**Build fails:**

- Check the build logs on Render
- Ensure `requirements.txt` has all dependencies
- Verify Python version in `runtime.txt`

**App crashes on startup:**

- Check if `loan_model.pkl` is in the repository
- Verify the Start Command is `gunicorn app:app`
- Check application logs on Render dashboard

**404 Error**:

- Ensure templates folder exists with `index.html`
- Check file paths are correct

**Slow first load:**

- This is normal on free tier after 15 min of inactivity
- App "wakes up" on first request

## Updating Your App

When you make changes:

```bash
git add .
git commit -m "Description of changes"
git push origin main
```

Render will automatically detect the push and redeploy!

## Custom Domain (Optional)

1. Go to your service settings on Render
2. Click "Custom Domain"
3. Follow instructions to add your domain
4. Update DNS settings with your domain provider

## Monitoring

- View logs: Render Dashboard → Your Service → Logs
- Check metrics: CPU, Memory usage in dashboard
- Set up health checks in service settings

## Cost

- **Free Tier**: $0/month
  - 750 hours/month free
  - Spins down after 15 min inactivity
  - Perfect for testing and demos

- **Paid Tier**: Starting at $7/month
  - Always on
  - Better performance
  - No spin down

## Support

If you encounter issues:

1. Check Render documentation: <https://render.com/docs>
2. Check application logs on Render
3. GitHub Issues for Flask/Render specific problems

---

## Quick Checklist

Before deploying, make sure:

- [ ] `loan_model.pkl` is in the project folder
- [ ] All files are committed to Git
- [ ] Repository is pushed to GitHub
- [ ] GitHub repository is public (for free tier)
- [ ] All configuration files are present

Good luck with your deployment! 🚀
