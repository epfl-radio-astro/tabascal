site_configuration = {
    "systems": [
        {
            "name": "daint",
            "descr": "Piz Daint",
            "hostnames": [".*"],
            "partitions": [
                {
                    "name": "gpu",
                    "descr": "GPU partition",
                    "scheduler": "local",
                    "launcher": "local",
                    "environs": ["builtin"],
                },
            ],
        },
    ],
    "environments": [
        {
            "name": "builtin",
            "cc": "cc",
            "cxx": "CC",
        },
    ],
}
