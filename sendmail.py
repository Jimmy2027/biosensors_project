import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.application import MIMEApplication
from email import encoders

def sendmail(kernel_size,save_dir,model_dir,img1_path, img2_path):
    email_user = 'hendrik.klug@gmail.com'
    strangething = ''
    email_send = 'hendrik.klug@gmail.com'

    subject = 'Sensors_project data'

    msg = MIMEMultipart()
    msg['From'] = email_user
    msg['To'] = email_send
    msg['Subject'] = subject

    body = 'Job finished  \n'+'Kernel Size: '+str(kernel_size)+' \n \n \n \n \n '
    msg.attach(MIMEText(body,'plain'))

    files = [model_dir, img1_path, img2_path]

    for f in files:  # add files to the message

        attachment = open(os.path.join(save_dir,f), 'rb')
        part = MIMEBase('application', 'octet-stream')
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment', filename=f+'\n\n \n \n')
        msg.attach(part)



    text = msg.as_string()
    server = smtplib.SMTP('smtp.gmail.com',587)
    server.starttls()
    server.login(email_user,strangething)


    server.sendmail(email_user,email_send,text)
    print('mail sent!')
    server.quit()
