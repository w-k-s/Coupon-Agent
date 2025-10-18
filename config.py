from dotenv import dotenv_values


class Config:
    def __init__(self, env_file=".env"):
        env_vars = dotenv_values(env_file, verbose=True)
        for key, value in env_vars.items():
            setattr(self, key, value)

        assert self.LANGSMITH_TRACING is not None
        assert self.LANGSMITH_API_KEY is not None
        assert self.LANGSMITH_PROJECT is not None
        assert self.OPENAI_API_KEY is not None
        assert self.COUPON_DB_CONN_STRING is not None
        assert self.CHECKPOINT_DB_CONN_STRING is not None
