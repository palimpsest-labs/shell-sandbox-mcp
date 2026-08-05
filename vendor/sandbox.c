/*
 * sandbox — pledge+unveil-based command sandbox
 *
 * Usage: sandbox PROMISES UNVEIL_DIR -- cmd [args...]
 *
 * Pledges the given promises, unveils UNVEIL_DIR for rwc access,
 * then execs cmd with args.
 *
 * Optional env: SANDBOX_UNVEIL_R — a colon-separated list of extra paths
 * to unveil read-only (e.g. git config dotfiles under $HOME).
 * SANDBOX_UNVEIL_RW — extra paths to unveil read-write (e.g. python's
 * user site-packages and pip/uv caches).
 *
 * Compile with cosmocc for a portable static binary:
 *   cosmocc -o sandbox sandbox.c
 */

#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* pledge() and unveil() are available on Linux (including Cosmopolitan) */
extern int pledge(const char *, const char *);
extern int unveil(const char *, const char *);

/*
 * Unveil each colon-separated path from env var `name` with the given
 * permissions. Missing paths are tolerated (ENOENT), since optional dirs
 * may not exist yet. Returns 0 on success, -1 on error.
 */
static int unveil_list_from_env(const char *name, const char *perms) {
    const char *value = getenv(name);
    if (!value || !*value)
        return 0;

    char *copy = strdup(value);
    if (!copy) {
        fprintf(stderr, "sandbox: out of memory\n");
        return -1;
    }
    int rc = 0;
    char *save = NULL;
    for (char *tok = strtok_r(copy, ":", &save); tok; tok = strtok_r(NULL, ":", &save)) {
        if (*tok && unveil(tok, perms) != 0 && errno != ENOENT) {
            fprintf(stderr, "sandbox: unveil('%s', '%s') failed: %s\n",
                    tok, perms, strerror(errno));
            rc = -1;
            break;
        }
    }
    free(copy);
    return rc;
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        fprintf(stderr, "Usage: sandbox PROMISES UNVEIL_DIR -- cmd [args...]\n");
        fprintf(stderr, "Example: sandbox 'stdio rpath' /tmp -- cat /etc/hostname\n");
        return 1;
    }

    const char *promises = argv[1];
    const char *unveil_dir = argv[2];

    /* Find the '--' separator */
    int sep = 0;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--") == 0) {
            sep = i;
            break;
        }
    }
    if (sep == 0) {
        fprintf(stderr, "sandbox: missing '--' separator before command\n");
        return 1;
    }

    char *cmd = argv[sep + 1];
    if (!cmd) {
        fprintf(stderr, "sandbox: no command after '--'\n");
        return 1;
    }

    /* Unveil: restrict filesystem to working dir, binary paths, and /tmp */
    if (unveil(unveil_dir, "rwc") != 0) {
        fprintf(stderr, "sandbox: unveil('%s', 'rwc') failed: %s\n",
                unveil_dir, strerror(errno));
        return 1;
    }
    /* Allow reading+executing binaries and libraries */
    unveil("/usr/bin", "rx");
    unveil("/usr/lib", "rx");
    unveil("/lib", "rx");
    unveil("/lib64", "rx");
    unveil("/usr/local/bin", "rx");
    unveil("/bin", "rx");
    unveil("/tmp", "rwc");
    unveil("/etc", "r");  /* git needs /etc for config */
    /* file(1) magic database; curl needs DNS resolver config in /run
       (/etc/resolv.conf is a symlink to /run/systemd/resolve/stub-resolv.conf) */
    unveil("/usr/share", "r");
    unveil("/run", "r");
    /* Devices: git, cargo, and shell tools need /dev/null and /dev/urandom */
    unveil("/dev/null", "rw");
    unveil("/dev/urandom", "r");
    /* Also unveil the binary itself if we can determine its path */
    if (cmd[0] == '/') {
        unveil(cmd, "rx");
    }
    /* Additional read-only paths from SANDBOX_UNVEIL_R (colon-separated).
       Used to let commands like git read config dotfiles under $HOME that
       the default unveil set doesn't cover. Paths may not exist yet (unveil
       on a missing path is a no-op, which is fine for optional config). */
    if (unveil_list_from_env("SANDBOX_UNVEIL_R", "r") != 0)
        return 1;
    /* Additional read-write paths from SANDBOX_UNVEIL_RW (colon-separated),
       e.g. python's user site-packages and pip/uv caches so packages can be
       installed and used from inside the sandbox. */
    if (unveil_list_from_env("SANDBOX_UNVEIL_RW", "rwc") != 0)
        return 1;
    /* Lock unveil — no further paths can be added */
    if (unveil(NULL, NULL) != 0) {
        fprintf(stderr, "sandbox: unveil lock failed: %s\n", strerror(errno));
        return 1;
    }

    /* Apply the pledge — always include exec so we can execvp() */
    char full_promises[512];
    int wrote = snprintf(full_promises, sizeof(full_promises), "%s exec", promises);
    if (wrote < 0 || (size_t)wrote >= sizeof(full_promises)) {
        fprintf(stderr, "sandbox: promise string too long (%d chars, max %zu)\n",
                wrote, sizeof(full_promises) - 1);
        return 1;
    }
    if (pledge(full_promises, NULL) != 0) {
        fprintf(stderr, "sandbox: pledge('%s') failed: %s\n",
                full_promises, strerror(errno));
        return 1;
    }

    /* Exec the command */
    execvp(cmd, &argv[sep + 1]);

    /* execvp only returns on error */
    fprintf(stderr, "sandbox: exec '%s' failed: %s\n", cmd, strerror(errno));
    return 1;
}
