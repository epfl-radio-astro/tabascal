site_configuration = {
    "systems": [
        {
            "name": "daint",
            "descr": "Piz Daint",
            "hostnames": ["daint"],
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
        {
            "name": "generic",
            "descr": "Generic local system",
            "hostnames": [".*"],
            "partitions": [
                {
                    "name": "default",
                    "descr": "Default partition",
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
