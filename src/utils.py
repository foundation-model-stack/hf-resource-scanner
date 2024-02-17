# adapted from https://stackoverflow.com/a/1094933
def fmt_size(size):
    """ Converts size in bytes to human readable value."""

    for unit in ("", "K", "M", "G", "T"):
        if abs(size) < 1024.0:
            return f"{size:3.1f} {unit}B"
        size /= 1024.0

    return f"{size:.1f} PB"
