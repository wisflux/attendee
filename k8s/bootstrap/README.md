# Bootstrap

Resources here are **applied by hand, once, when standing up a namespace**. ArgoCD does
not track this directory — no Application points at it, and nothing in CI applies it.

That is the entire point. Everything in here either holds a real secret value or is a
one-shot operation, so re-applying it on every deploy is at best pointless and at worst
destructive. Previously these lived under `k8s/app`, which ArgoCD re-synced on every
image-tag push — meaning the placeholder secrets in git overwrote the real ones in the
cluster on every deploy.

## Layout

| Path | Why it is not in `k8s/app` |
|---|---|
| `namespace.yaml` | Created once; the Applications set `CreateNamespace=false`. |
| `rbac.yaml` | ServiceAccount + Role the app uses to launch bot pods. Changes rarely. |
| `namespace-admin.yaml` | Admin SA whose token backs the kubeconfig. Gitignored. |
| `secrets/*.example.yaml` | Templates. The real `*.yaml` are gitignored and live only in the cluster. |
| `jobs/` | One-shot init Jobs. Job specs are immutable — re-applying a changed one fails. |

## Standing up a fresh namespace

Run in order; each step depends on the previous.

```sh
# 1. Namespace
kubectl apply -f k8s/bootstrap/namespace.yaml

# 2. Secrets — copy each example, fill in real values, then apply.
#    See the header comment in each file for how to generate values.
cp k8s/bootstrap/secrets/attendee-secret.example.yaml k8s/bootstrap/secrets/attendee-secret.yaml
cp k8s/bootstrap/secrets/postgres-secret.example.yaml k8s/bootstrap/secrets/postgres-secret.yaml
cp k8s/bootstrap/secrets/minio-secret.example.yaml    k8s/bootstrap/secrets/minio-secret.yaml
$EDITOR k8s/bootstrap/secrets/*.yaml
kubectl apply -f k8s/bootstrap/secrets/attendee-secret.yaml \
               -f k8s/bootstrap/secrets/postgres-secret.yaml \
               -f k8s/bootstrap/secrets/minio-secret.yaml

# 3. Image pull secret — generated, not hand-written.
#    See secrets/regcred.example.yaml for the command.

# 4. RBAC
kubectl apply -f k8s/bootstrap/rbac.yaml
kubectl apply -f k8s/bootstrap/namespace-admin.yaml

# 5. Platform (Postgres/MinIO/Redis/ingress) — sync the attendee-platform
#    ArgoCD Application, then wait for Postgres and MinIO to be Ready.
kubectl -n meeting-utility rollout status deploy/postgres deploy/minio

# 6. Init jobs — only after Postgres and MinIO are up.
kubectl apply -f k8s/bootstrap/jobs/

# 7. App — sync the attendee ArgoCD Application.
```

## Rotating a secret

Edit the real (gitignored) file and re-apply, then restart consumers — pods read env
from Secrets at start and will not pick up changes on their own:

```sh
kubectl apply -f k8s/bootstrap/secrets/attendee-secret.yaml
kubectl -n meeting-utility rollout restart deploy/attendee-web deploy/attendee-worker \
        deploy/attendee-scheduler deploy/attendee-webpage-streamer
```

Changing `postgres-secret` or `minio-secret` after data exists does **not** re-key the
running Postgres/MinIO — their passwords are set at first init from the PVC. Rotating
those means changing the credential inside the service too.

## Re-running an init job

Jobs are immutable. Delete before re-applying:

```sh
kubectl -n meeting-utility delete job postgres-init-db minio-init-buckets --ignore-not-found
kubectl apply -f k8s/bootstrap/jobs/
```

## Longer term

The migration to the Vault-equipped cluster should replace `secrets/` with External
Secrets Operator `ExternalSecret` resources. Those hold only Vault *references*, not
values, so they can live in `k8s/app` and sync freely — which removes the manual step
this directory exists to protect.
