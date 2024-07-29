from django.db import models

class TrialData(models.Model):
    address = models.CharField(max_length=255)
    paradigm = models.CharField(max_length=100)
    modality = models.CharField(max_length=50)
    date = models.DateTimeField(null=True, blank=True)
    subject = models.CharField(max_length=100)
    label = models.CharField(max_length=50)
    trialid = models.CharField(max_length=50)
    trialsample = models.IntegerField()
    trialsfreq = models.IntegerField()
    # 由于Django模型字段并不直接支持numpy数组，你可以将其保存为文件路径或使用TextField存储序列化数据
    trialchan = models.TextField()
    sexual = models.CharField(max_length=1)
    age = models.IntegerField()
    scibackground = models.IntegerField()
    sleephours = models.FloatField()
    sleepqaulity = models.FloatField()
    brainhealth = models.CharField(max_length=100)
    workout = models.FloatField()
    exptime = models.FloatField()
    trialbaseline = models.IntegerField()
    # 文件路径字段，用来关联.npy文件
    npy_file_path = models.CharField(max_length=255, blank=True, null=True)
