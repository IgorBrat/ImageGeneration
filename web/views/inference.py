from django.shortcuts import render

from web.utils import load_presaved_model, tensors_to_base64


def infer(request, model_name, version):
    context = {}
    model = load_presaved_model(model_name, version)
    if request.method == 'POST':
        # General
        num_images = int(request.POST.get('num_images'))
        # Text2Image
        query = request.POST.get('query')
        # CondGAN
        target_class = request.POST.get('target_class')
        # ControlGAN
        feature = request.POST.get('feature')
        # StyleGAN
        num_of_steps = request.POST.get('num_steps')
        alpha = request.POST.get('alpha')
        if query:
            images = model.generate(num=num_images, query=query)
        elif target_class:
            images = model.generate(num=num_images, target_class=int(target_class))
        elif feature:
            images = model.generate(num=num_images, feature=int(feature))
        elif num_of_steps:
            images = model.generate(num_images=num_images, num_steps=int(num_of_steps), alpha=float(alpha))
        else:
            images = model.generate(num=num_images)
        images = tensors_to_base64(images)
        context['images'] = images
    return render(request, fr'inference/models/{model_name}.html', context)
