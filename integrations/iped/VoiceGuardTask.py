# VoiceGuard audio-deepfake detection task for IPED (https://github.com/sepinf-inc/IPED).
#
# IPED ships speech-TO-TEXT tasks (WhisperProcess / Wav2Vec2Process) but has NO
# audio-deepfake detector. This task adds that capability: for every audio item it
# asks a running VoiceGuard server whether the voice is genuine or AI-generated and
# records the verdict as searchable columns + a category + a bookmark.
#
# Design: the task is a thin client over the VoiceGuard HTTP API (stdlib `urllib`
# only — no torch/CUDA inside IPED's jep threads), so it never competes with IPED
# for the GPU. The VoiceGuard server (which runs the deployed SSL model) can run on
# the same host, even air-gapped.
#
# Install: see integrations/iped/README.md (copy to <IPED>/scripts/tasks/, register
# in conf/TaskInstaller.xml, run the VoiceGuard server).
#
# The class name MUST equal the file name (IPED convention).

import json
import os
import urllib.error
import urllib.parse
import urllib.request

AUDIO_EXTS = {"wav", "mp3", "flac", "m4a", "ogg", "opus", "aac", "wma", "amr", "3gp"}


class VoiceGuardTask:
    # ---- configuration (env-overridable; safe defaults for a local server) -------
    def _cfg(self):
        return {
            "enabled": os.environ.get("VOICEGUARD_IPED_ENABLED", "true").lower() == "true",
            "api": os.environ.get("VOICEGUARD_API", "http://127.0.0.1:8000").rstrip("/"),
            "user": os.environ.get("VOICEGUARD_USER", "admin"),
            "password": os.environ.get("VOICEGUARD_PASSWORD", "voiceguard2026"),
            "timeout": float(os.environ.get("VOICEGUARD_TIMEOUT", "30")),
            "logfile": os.environ.get("VOICEGUARD_IPED_LOG", ""),
        }

    def _log(self, msg):
        # Optional audit log (jep can swallow Python print); set VOICEGUARD_IPED_LOG.
        lf = getattr(self, "conf", {}).get("logfile", "")
        if not lf:
            return
        try:
            with open(lf, "a") as f:
                f.write(msg + "\n")
        except Exception:
            pass

    def isEnabled(self):
        return self._cfg()["enabled"]

    def getConfigurables(self):
        return []

    def init(self, configuration):
        self.conf = self._cfg()
        self.token = None
        self.flagged = 0
        try:
            self._authenticate()
        except Exception as e:
            # Don't fail init — process() will retry auth lazily and degrade gracefully.
            print("[VoiceGuard] init: auth deferred (%s)" % str(e)[:160])

    # ---- HTTP helpers (stdlib only) ----------------------------------------------
    def _authenticate(self):
        data = urllib.parse.urlencode(
            {"username": self.conf["user"], "password": self.conf["password"]}
        ).encode()
        # Resolve the API root: works whether the user runs the bare API
        # (uvicorn voiceguard.api.main:app -> /token) or the combined demo server
        # (mounts the API under /api -> /api/token). Lock in whichever responds.
        base = self.conf["api"]
        candidates = [base] if base.endswith("/api") else [base, base + "/api"]
        last = None
        for cand in candidates:
            req = urllib.request.Request(
                cand + "/token",
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            try:
                with urllib.request.urlopen(req, timeout=self.conf["timeout"]) as r:
                    self.token = json.loads(r.read().decode())["access_token"]
                self.conf["api"] = cand  # remember the working base for /detect
                return
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (404, 405):
                    continue  # wrong layout -> try /api variant
                raise
        raise last

    def _multipart(self, file_path):
        boundary = "----VoiceGuardIPEDBoundary7f3a"
        name = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            content = f.read()
        head = (
            "--%s\r\n" % boundary
            + 'Content-Disposition: form-data; name="file"; filename="%s"\r\n' % name
            + "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        tail = ("\r\n--%s--\r\n" % boundary).encode()
        return b"".join([head, content, tail]), boundary

    def _detect(self, file_path):
        """POST the audio to /detect; re-auth once on 401. Returns parsed dict."""
        body, boundary = self._multipart(file_path)
        for attempt in (1, 2):
            if not self.token:
                self._authenticate()
            req = urllib.request.Request(
                self.conf["api"] + "/detect",
                data=body,
                headers={
                    "Content-Type": "multipart/form-data; boundary=%s" % boundary,
                    "Authorization": "Bearer %s" % self.token,
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=self.conf["timeout"]) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as he:
                if he.code == 401 and attempt == 1:
                    self.token = None  # token expired on a long case -> re-auth + retry
                    continue
                raise
        return None

    def _is_audio(self, item):
        mt = item.getMediaType()
        if mt is not None and str(mt.toString()).lower().startswith("audio"):
            return True
        ext = item.getExt()
        return ext is not None and ext.lower() in AUDIO_EXTS

    # ---- main entry: MUST NOT raise (a throw can abort the whole IPED job) --------
    def process(self, item):
        try:
            if item.isDir() or item.getLength() is None or item.getLength() == 0:
                return
            if not self._is_audio(item):
                return
            tmp = item.getTempFile()
            if tmp is None:
                return
            result = self._detect(tmp.getAbsolutePath())
            if not result:
                item.setExtraAttribute("voiceguard:error", "no response")
                return
            label = str(result.get("label", "unknown"))
            conf = float(result.get("confidence", 0.0))
            fake_prob = conf if label == "fake" else (1.0 - conf)
            item.setExtraAttribute("voiceguard:deepfake", label)
            item.setExtraAttribute("voiceguard:fakeProbability", round(fake_prob, 4))
            item.setExtraAttribute("voiceguard:model", str(result.get("model", "")))
            if label == "fake":
                item.addCategory("Deepfake Audio (VoiceGuard)")
                self.flagged += 1
            self._log("%s -> %s (fakeProb=%.4f)" % (item.getName(), label, fake_prob))
        except Exception as e:  # never propagate into the IPED pipeline
            try:
                item.setExtraAttribute("voiceguard:error", str(e)[:200])
            except Exception:
                pass
            self._log("%s -> ERROR %s" % (item.getName(), str(e)[:160]))

    # ---- after the case: bookmark every suspected deepfake -----------------------
    # NB: IPED runs finish() once per worker (separate jep interpreters/processes),
    # so this is called several times; IPED keys the bookmark by name, so the items
    # land in a single "VoiceGuard - Suspected Deepfake Audio" bookmark regardless.
    def finish(self):
        try:
            searcher.setQuery("voiceguard\\:deepfake:fake")
            ids = searcher.search().getIds()
            if ids is not None and len(ids) > 0:
                bid = ipedCase.getBookmarks().newBookmark("VoiceGuard - Suspected Deepfake Audio")
                ipedCase.getBookmarks().setBookmarkComment(
                    bid, "Audio flagged AI-generated by the VoiceGuard detector."
                )
                ipedCase.getBookmarks().addBookmark(ids, bid)
                ipedCase.getBookmarks().saveState(True)
                self._log("bookmarked %d suspected-deepfake item(s)" % len(ids))
            else:
                self._log("finish: no items matched voiceguard:deepfake:fake")
        except Exception as e:
            self._log("finish: bookmark step skipped (%s)" % str(e)[:160])
