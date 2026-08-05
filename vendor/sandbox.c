/*
 * sandbox — pledge+unveil-based command sandbox
 *
 * Usage: sandbox PROMISES UNVEIL_DIR -- cmd [args...]
 *
 * Pledges the given promises, unveils UNVEIL_DIR for rwc access,
 * then execs cmd with args.
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

int main(int argc, char *argv[]) {
    if (argc < 6) {
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
    /* Unveil minimal /dev entries: git and other tools need /dev/null and
     * /dev/urandom (random). Keep the surface as tight as possible. */
    unveil("/dev/null", "rw");
    unveil("/dev/urandom", "r");
    /* Git reads global config from the user's home dir (~/.gitconfig and
     * ~/.config/git/config). Unveil just those files, not the whole home. */
    const char *home = getenv("HOME");
    if (home && home[0]) {
        size_t home_len = strlen(home);
        if (home_len < 1024) {
            char gitconfig[1100];
            snprintf(gitconfig, sizeof(gitconfig), "%s/.gitconfig", home);
            unveil(gitconfig, "r");
            char xdg_config[1100];
            snprintf(xdg_config, sizeof(xdg_config), "%s/.config/git/config", home);
            unveil(xdg_config, "r");
        }
    }
    /* Also unveil the binary itself if we can determine its path */
    if (cmd[0] == '/') {
        unveil(cmd, "rx");
    }
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
