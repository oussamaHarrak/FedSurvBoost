from importlib import import_module
from os.path import splitext

from .random_grouped_assigner import RandomGroupedAssigner


class AdaBoostAssigner(RandomGroupedAssigner):

    def __init__(self, task_groups, **kwargs):
        super().__init__(task_groups, **kwargs)

    def get_aggregation_type_for_task(self, task_name):
        """Extract aggregation type from self.tasks."""
        if 'aggregation_type' not in self.tasks[task_name]:
            return None

        template = self.tasks[task_name]['aggregation_type']
        # TODO: this should be integrated better with aggregator (and the eventual parameter should be handled better)
        if isinstance(template, str):
            class_name = splitext(template)[1].strip('.')
            module_path = splitext(template)[0]
            module = import_module(module_path)
            if 'n_classes' in self.tasks[task_name]:
                template = getattr(module, class_name)(self.tasks[task_name]['n_classes'])
            else:
                template = getattr(module, class_name)()

        return template