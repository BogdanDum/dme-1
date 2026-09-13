"""Per-source parsers that turn Bronze archives into Silver tables.

Each module owns one supplied source, applies that source's checks, and stages
its own tables. Sources never read each other: integrating the three sources is
Gold's job, and joining them in Silver is explicitly not allowed.
"""
