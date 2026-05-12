# Project website (`docs/`)

A self-contained, single-page showcase for the TelecomAudit paper:
[`index.html`](index.html). It embeds the paper PDF, the three headline figures,
the artifact map, the deployment surfaces, and reproduction commands.

## Publishing

Two supported paths (either works):

1. **GitHub Actions (recommended, already wired):** the
   [`pages.yml`](../.github/workflows/pages.yml) workflow publishes this folder
   on every push to `main`. In the repo: *Settings → Pages → Build and
   deployment → Source: GitHub Actions*.

2. **Classic branch source:** *Settings → Pages → Source: Deploy from a branch →
   `main` / `/docs`*. The [`.nojekyll`](.nojekyll) file disables Jekyll so the
   static HTML and assets are served verbatim.

The site is then available at
`https://chimansalavati.github.io/telecomts-real-synthetic-gap/`.

## Assets

`assets/` holds copies of the paper PDF and the three figures referenced by the
page, so the site is fully self-contained and does not depend on files outside
`docs/`.
