COMPONENTS = {}

def register_component(name):
    def wrapper(cls):
        COMPONENTS[name] = cls
        return cls

    return wrapper