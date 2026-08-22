import os

from routes import app
import toolbox  # noqa: F401 - imported for its /mcp route registration

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), threaded=True)
