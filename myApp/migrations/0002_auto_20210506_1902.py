# Generated by Django 3.2 on 2021-05-06 13:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myApp', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='CustomModelFiles',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('files', models.FileField(upload_to='model/custom')),
            ],
        ),
        migrations.DeleteModel(
            name='Input',
        ),
    ]
