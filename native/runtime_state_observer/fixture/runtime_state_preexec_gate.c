#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
    char command[32];
    if (argc != 2) {
        fprintf(stderr, "usage: %s TARGET\n", argv[0]);
        return 2;
    }
    printf("GATED %ld\n", (long)getpid());
    fflush(stdout);
    if (fgets(command, sizeof(command), stdin) == NULL) {
        return 1;
    }
    if (strcmp(command, "exit\n") == 0) {
        return 0;
    }
    if (strcmp(command, "exec\n") != 0) {
        fputs("pre-exec gate received an invalid command\n", stderr);
        return 2;
    }
    execl(argv[1], argv[1], (char *)NULL);
    fprintf(stderr, "pre-exec gate failed: %s\n", strerror(errno));
    return 1;
}
