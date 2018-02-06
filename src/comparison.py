import evaluation
from dataloader import CustomDataloader



def evaluate_with_comparison(networks, dataloader, **options):
    comparison_dataloader = get_comparison_dataloader(**options)
    if comparison_dataloader:
        options['fold'] = 'openset_{}'.format(comparison_dataloader.dsf.name)

    new_results = evaluation.evaluate_classifier(networks, dataloader, comparison_dataloader, **options)

    if comparison_dataloader:
        openset_results = evaluation.evaluate_openset(networks, dataloader, comparison_dataloader, **options)
        new_results[options['fold']].update(openset_results)
    return new_results


def get_comparison_dataloader(comparison_dataset=None, **options):
    if not comparison_dataset:
        return
    comparison_options = options.copy()
    comparison_options['dataset'] = comparison_dataset
    comparison_options['last_batch'] = True
    comparison_options['shuffle'] = False
    comparison_dataloader = CustomDataloader(**comparison_options)
    return comparison_dataloader
