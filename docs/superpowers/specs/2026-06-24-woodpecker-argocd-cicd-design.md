# Woodpecker CI + Argo CD Pipeline Design

## Overview

Set up a CI/CD pipeline using Woodpecker CI and Argo CD for the Attendee application. Both tools are already running in the same cluster (meeting-utility namespace at 91.98.183.108).

## Trigger

**On push to `prod` branch only.** The `main` branch is the working branch (receives upstream pulls and feature work). Merging `main` into `prod` is the explicit deployment gate.

## Pipeline Steps

### 1. Build Docker Image

- Build from the existing `Dockerfile` at repo root
- Platform: `linux/amd64`

### 2. Push to Docker Hub

- Registry: Docker Hub
- Image: `wisfluxp/attendee:<commit-sha>`
- Credentials: Docker Hub username + PAT (stored as Woodpecker secrets)

### 3. Update K8s Manifests

Update the image tag in all deployment files that reference `wisfluxp/attendee`:

- `k8s/app/web/deployment.yaml`
- `k8s/app/worker/deployment.yaml`
- `k8s/app/scheduler/deployment.yaml`
- `k8s/app/webpage-streamer/deployment.yaml`
- `k8s/app/janitor/cronjob.yaml`

Replace the image value (currently `wisfluxp/attendee:latest`) with `wisfluxp/attendee:<commit-sha>`.

### 4. Commit & Push Manifests

- Commit the updated manifests back to the `prod` branch
- Commit message: `ci: update image tag to <commit-sha>`
- Use a bot/service account for the git push
- The pipeline must not re-trigger itself on this commit (Woodpecker's `[CI SKIP]` or branch event filtering)

### 5. Trigger Argo CD Sync

- Call Argo CD API to sync the application
- Argo CD Application watches the `prod` branch, path `k8s/`
- Endpoint: Argo CD server URL within the cluster
- Auth: Argo CD API token (stored as Woodpecker secret)

## Argo CD Application

A new `argocd/application.yaml` manifest (kept outside `k8s/` to avoid Argo CD managing its own Application resource):

- **Source:** This GitHub repo, `prod` branch, path `k8s/`
- **Destination:** Local cluster, `meeting-utility` namespace
- **Sync policy:** Manual (triggered by Woodpecker pipeline)
- **Self-heal:** Disabled (we want git to be the source of truth)
- **Applied manually once:** `kubectl apply -f argocd/application.yaml` — not managed by Argo CD itself

## Secrets Required in Woodpecker

| Secret | Purpose |
|--------|---------|
| `docker_username` | Docker Hub username (`wisfluxp`) |
| `docker_password` | Docker Hub PAT |
| `argocd_server` | Argo CD server URL (cluster-internal) |
| `argocd_token` | Argo CD API auth token |
| `git_push_token` | GitHub PAT for pushing manifest commits back to `prod` |

## Re-trigger Prevention

When Woodpecker commits updated manifests back to `prod`, the pipeline must not re-trigger. Options:
- Include `[CI SKIP]` in the commit message
- Use Woodpecker's `when` conditions to filter out commits by the bot user

## Files to Create/Modify

| File | Action |
|------|--------|
| `.woodpecker.yml` | Create — full pipeline config |
| `argocd/application.yaml` | Create — Argo CD Application manifest (outside k8s/ to avoid self-management) |
| `k8s/app/*/deployment.yaml` | Modify — change `latest` tag to a placeholder or leave as-is (pipeline updates it) |

## What Stays Unchanged

- GitHub Actions workflows (`.github/workflows/`) — kept for upstream compatibility
- Existing K8s manifests structure — no restructuring needed
- Docker Hub as the registry — already in use

## Architecture Diagram

```
push to prod
    │
    ▼
Woodpecker CI
    │
    ├── 1. Build image
    ├── 2. Push wisfluxp/attendee:<sha> to Docker Hub
    ├── 3. Update image tags in k8s/app/*/deployment.yaml
    ├── 4. Commit + push to prod [CI SKIP]
    └── 5. POST to Argo CD API → sync
                                    │
                                    ▼
                              Argo CD syncs
                              k8s/ manifests
                              to cluster
```
