import os
import time
import cPickle
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import string
import pandas as pd
from PIL import Image
import cStringIO as StringIO
import urllib
#import exifutil

from sketcher import *
from PCAProcess import *
from photo import *
REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../..')
#LOG_FOLDER = '/tmp/caffe_demos_uploads'
LOG_FOLDER = os.getcwd()+'/log'
UPLOAD_FOLDER = os.getcwd()+'/imageresdatabase'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

# Obtain the flask app object
app = flask.Flask(__name__)

@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)


@app.route('/photo2sketch', methods=['GET', 'POST'])
def photo2sketch():
    if flask.request.method == 'GET':
        return flask.redirect('/')

    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(LOG_FOLDER, filename_)
        imagefile.save(filename)
        #save the image to be processed in the log directory
        original_image = app.sketcher.origin(filename)
        crop_image = app.sketcher.crop(filename)
        output_image = app.sketcher.process(crop_image)


    except Exception as err:
        #logging.info('Uploaded image open error: %s', err)
        logging.info('error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    return flask.render_template('photo2sketch.html', has_result=True, 
        input=embed_image_html(original_image), 
        result=embed_image_html(output_image))

    #return flask.render_template('index.html', has_result=False)
    #return flask.render_template(
    #    'index.html', has_result=True, result=result,
    #    imagesrc=embed_image_html(image)
    #)

@app.route('/sketch2photo', methods=['GET', 'POST'])
def sketch2photo():
    if flask.request.method == 'GET':
        return flask.redirect('/')

    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(LOG_FOLDER, filename_)
        imagefile.save(filename)

        original_image = app.photo.origin(filename)
        crop_image = app.photo.crop(filename)

        output_image = app.photo.process(crop_image)
        #crop_image = app.sketcher.crop(filename)
        #output_image = app.sketcher.process(crop_image)

    except Exception as err:
        #logging.info('Uploaded image open error: %s', err)
        logging.info('error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    return flask.render_template('photo2sketch.html', has_result=True, 
        input=embed_image_html(original_image), 
        result=embed_image_html(output_image))

    #return flask.render_template('index.html', has_result=False)
    #return flask.render_template(
    #    'index.html', has_result=True, result=result,
    #    imagesrc=embed_image_html(image)
    #)

def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    #image_pil = image.fromarray((255 * image).astype('uint8'))
    #image_pil = image.resize((167, 256))
    string_buf = StringIO.StringIO()
    image.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data

def embed_image_html2(image):
    """Creates an image embedded in HTML base64 format."""
    #image_pil = Image.fromarray((255 * image).astype('uint8'))
    image_pil = image.resize((150, 187))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data

@app.route('/searchphoto', methods=['GET', 'POST'])
def searchphoto():
    if flask.request.method == 'GET':
        return flask.redirect('/')

    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(LOG_FOLDER, filename_)
        imagefile.save(filename)

        crop_image =Image.open(filename)
        
        
        imgList = match(crop_image)
        if crop_image.size[0]!=143 and crop_image.size[1]!=188:
            crop_image = Image.open("./templates/error.jpg")
 
        oimgList = []
        for x in imgList:
            oimgList.append(Image.open(x))

        os.remove(filename)

    except Exception as err:
        #logging.info('Uploaded image open error: %s', err)
        logging.info('error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    return flask.render_template('index.html', has_result=True, 
        input=embed_image_html(crop_image), 
        result1=embed_image_html2(oimgList[0]), 
        result2=embed_image_html2(oimgList[1]), 
        result3=embed_image_html2(oimgList[2]),
        result4=embed_image_html2(oimgList[3]),
        result5=embed_image_html2(oimgList[4]),
        result6=embed_image_html2(oimgList[5]))
    #return flask.render_template('index.html', has_result=False)
    #return flask.render_template(
    #    'index.html', has_result=True, result=result,
    #    imagesrc=embed_image_html(image)
    #)
@app.route('/uploadphoto', methods=['GET', 'POST'])
def uploadphoto():
    if flask.request.method == 'GET':
        return flask.redirect('/')
    imagefile = flask.request.files.getlist('imagefile')
    total = 0.0;
    success = 0.0;
    logname = str(datetime.datetime.now()).replace(' ', '_')+".txt"
    #print str(os.path.join((LOG_FOLDER,logname)))
    fd = open(UPLOAD_FOLDER+'/log/'+logname,'w+')
    for _file in imagefile:
        total += 1
        try:
            filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
                werkzeug.secure_filename(_file.filename)
            filename = os.path.join(UPLOAD_FOLDER, _file.filename)
            logname = os.path.join(LOG_FOLDER,filename_)
            
            _file.save(logname)
            print _file.filename
            writeNewImgPath(_file.filename)

            original_image = app.sketcher.origin(logname)
            crop_image = app.sketcher.crop(logname)
            output_image = app.sketcher.process(crop_image)
            output_image.show()
            
            crop_image.save(filename)
            output_image.save(logname)
            uploadnew(output_image)

            
            fd.write(_file.filename+" Succeed Uploaded "+ \
                str(datetime.datetime.now()).replace(' ', '_')+'\n')
            success += 1

        except Exception as err:
            #logging.info('Uploaded image open error: %s', err)
            #logging.info('error: %s', err)
            fd.write(_file.filename+ " Failed "+str(err)+' '+ \
               str(datetime.datetime.now()).replace(' ', '_')+'\n')            
            continue
            return flask.render_template(
                'index.html', has_result=True,
                result=(False, 'Cannot open uploaded image.')
            )
    
    fd.write("Total: "+ str(total) +" Success: "+str(success)+ \
        " Accurate Rate: "+str(success/total)+'\n')
    fd.close();        
    return flask.render_template('index.html', has_result=True, 
        input=embed_image_html(output_image))
    #return flask.render_template('index.html', has_result=False)
    #return flask.render_template(
    #    'index.html', has_result=True, result=result,
    #    imagesrc=embed_image_html(image)
    #)


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    if os.path.exists(UPLOAD_FOLDER+'/log'):
        print "File Exists"
    else:
        os.mkdir(UPLOAD_FOLDER+'/log')
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)

    opts, args = parser.parse_args()

    # Initialize classifier + warm start by forward for allocation
    app.sketcher = Sketcher()
    app.photo = Photo()
    #direct debug
    #filename = "./uploads/test.jpg"
    #crop_image = app.sketcher.crop(filename)
    #output_image = app.sketcher.process(crop_image)

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(LOG_FOLDER):
        os.makedirs(LOG_FOLDER)
    start_from_terminal(app)
