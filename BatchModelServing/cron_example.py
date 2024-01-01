from crontab import CronTab

cron = CronTab(user=True)
job = cron.new(command='cd /Users/quangtn/Desktop/01_work/'
                       '01_job/02_ml/bentoml/chapter6 && $(which python3) model.py')
job.minute.every(1)
# cron.remove_all() ## Command to remove all the scheduled jobs
cron.write()