# Changesets

Every change that a user of the package would notice gets a changeset: run `npx changeset`, pick
the bump (patch, minor, major) and write one or two sentences for the changelog. The file it
creates in this folder is committed with the change. On release, `npm run version-packages` turns the
pending changesets into a version bump and `CHANGELOG.md` entries; the release workflow then
publishes. See https://github.com/changesets/changesets.
