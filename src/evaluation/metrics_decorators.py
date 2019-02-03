import functools
import inspect


def handles_prediction(f):
    """
    Decorator for function that handles predictions, and may produce temporary info based on it.
    Function is assumed to have arguments: 'labels' and 'preds'.
    The function invocation will generate temporary data stored within an 'extent' attribute of the encapsulating
    object.
    The 'extent' attribute will be cleaned by the end of the function invocation.
    """
    def _get_arg(f, args, kwargs, arg_name):
        # required argument is in kwargs
        if arg_name in kwargs:
            return kwargs[arg_name]
        else:  # required argument is in args
            argspec = inspect.getfullargspec(f)
            argument_index = argspec.args.index(arg_name)
            return args[argument_index]

    @functools.wraps(f)
    def wrapper(metrics_instance, *args, **kwargs):
        # Fetch extra information from arguments (they're either in kwargs or args)
        extent = {
            'epoch': _get_arg(f, args, kwargs, 'epoch'),
            'labels': _get_arg(f, args, kwargs, 'labels'),
            'preds': _get_arg(f, args, kwargs, 'preds'),
        }
        try:
            setattr(metrics_instance, 'extent', extent)
            ret = f(metrics_instance, *args, **kwargs)
        finally:
            delattr(metrics_instance, 'extent')
        return ret

    return wrapper


def requires_confusion_matrix(f):
    """
    Decorator for functions that require the calculation of the confusion matrix.
    The decorator will check if the confusion matrix have already been calculated and stored within the extent
    of the encapsulating class for the current invocation.
    If not, the confusion matrix will be lazily-calculated.
    """
    @functools.wraps(f)
    def wrapper(metrics_instance, *args, **kwargs):
        assert hasattr(metrics_instance, 'extent'), \
            'Invoking function must generate "extent" information. Make sure it\'s wrapped with @handles_prediction'
        if 'confusion_matrix' not in metrics_instance.extent:
            epoch = metrics_instance.extent['epoch']
            labels = metrics_instance.extent['labels']
            preds = metrics_instance.extent['preds']
            epoch_statistics = metrics_instance._fetch_epoch_statistics(epoch)
            metrics_instance._update_confusion_matrix(epoch_statistics, labels, preds)
            metrics_instance.extent['confusion_matrix'] = epoch_statistics['confusion_matrix']
        ret = f(metrics_instance, *args, **kwargs)
        return ret

    return wrapper


def metric(name):
    """
    Decorator for all metric functions.
    Metrics are invoked only when configured.
    """
    def _metric(f):
        @functools.wraps(f)
        def wrapper(metrics_instance, *args, **kwargs):
            if name in metrics_instance.metrics:
                ret = f(metrics_instance, *args, **kwargs)
                return ret
            else:
                return None

        return wrapper
    return _metric
