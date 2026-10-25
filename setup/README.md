# Setup

A static page that writes an anyserve `server.toml` from a form, together
with the matching `anyserve serve` command and the client and worker snippets
that connect to it. Only values that differ from the defaults are written, so
the file stays as short as the deployment needs. Nothing leaves the browser.

Open https://anyserve.github.io/anyserve.universe/setup/, or serve the directory locally:

```bash
python -m http.server --directory setup 8080
# http://127.0.0.1:8080/
```

`config.js` holds the generator; `node --test setup/config.test.js` checks it.
The field defaults and hints follow the
[configuration reference](https://github.com/anyserve/anyserve/blob/main/docs/src/configuration.md);
keep them in step when a key changes there.
