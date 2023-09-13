# Change to True for debugging purposes
debug = True


def error_handler(func):
    def inner_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if debug:
                raise e
            else:
                print(f"Exception Occurred: {str(e)}")
                print("Press Enter to Safely Exit Program")
                input()
                quit()
    return inner_function
