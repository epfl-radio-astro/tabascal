# Publishing the Documentation

These docs are published on [Read the Docs](https://readthedocs.org), which builds
them straight from the repository. Every push to an activated branch, and every new
release tag, triggers a build of that version; readers switch between versions with
the version flyout that Read the Docs injects into every page.

## Versions

Read the Docs treats each git branch or tag as a separate *version* with its own URL
under `https://tabascal.readthedocs.io/en/<version>/`.

| Version | Built from | Purpose |
| --- | --- | --- |
| `stable` | the highest `vX.Y.Z` tag | The default version — what visitors get from the bare project URL. |
| `latest` | the `main` branch | The current development docs, ahead of any release. |
| `vX.Y.Z` | that release tag | Frozen docs for a published version. |
| other branches | that branch | Long-lived work that needs its own published docs. |

Two consequences worth knowing:

- **A version is only buildable if `.readthedocs.yaml` exists on it.** Branches and
  tags created before that file was added cannot be built, so the first release
  tag to appear as a version must be cut after it landed on `main`.
- **`stable` only exists once there is a release tag.** Until the first `vX.Y.Z` tag
  is pushed, the default version has to be `latest`; see the setup checklist below.

The version shown in the sidebar comes from the installed package metadata
(`pyproject.toml`'s `version`). Branch builds append the branch name to it —
`0.0.1 (latest)` — since that version number is the release being worked towards
rather than one that has been published.

### Publishing a release version

1. Set `version` in `pyproject.toml` to the new release number and merge it.
2. Tag the merge commit `vX.Y.Z` and push the tag.
3. Activate the new version in the Read the Docs admin (**Versions**), unless an
   automation rule already activates tags. `stable` follows the highest tag
   automatically once it is active.

### Publishing another branch

Activate it under **Versions** in the project admin. The branch needs
`.readthedocs.yaml` in it — rebase it onto `main` first if it predates this setup.
Deactivating a version hides it from the flyout again; nothing in the repository
needs to change either way.

## One-time project setup

Only needed until the project exists on Read the Docs:

1. Sign in to <https://readthedocs.org> with the GitHub account that can administer
   `epfl-radio-astro/tabascal` and import the repository. The project slug decides
   the domain, so pick `tabascal` if it is free.
2. Under **Admin → Settings**, set the default version to `latest` (the `main`
   branch) — `stable` does not exist until the first release tag — and set the
   default branch to `main`.
3. Under **Admin → Settings**, enable *Build pull requests for this project* so
   documentation changes get a preview build on each pull request.
4. Under **Versions**, activate `main` (published as `latest`) and any other branch
   that should be published.
5. After the first `vX.Y.Z` tag is published, change the default version to
   `stable` so the bare project URL serves the latest release.
6. Add the resulting URL to the repository description and to `[project.urls]` in
   `pyproject.toml`.

## Building the docs locally

```bash
pixi run -e dev docs-build
```

Then open `docs/_build/html/index.html`. Read the Docs and the `Docs` GitHub
workflow both build with warnings treated as errors, so it is worth reproducing
that before pushing:

```bash
pixi run -e dev sphinx-build -b html -W --keep-going docs docs/_build/html
```
