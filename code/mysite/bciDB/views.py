

# Create your views here.
from django.shortcuts import render
from .models import TrialData

def trial_data_list(request):
    data = TrialData.objects.all()  # 查询所有记录
    return render(request, 'bciDB/trial_data_list.html', {'data': data})
