from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import cgi
import hashlib
from model import Model
from PIL import Image
import numpy as np
import io
import result

mlp = Model('mlp')
shallow = Model('shallow')
deep = Model('deep')

class MyHandler(BaseHTTPRequestHandler):

  def _set_headers(self, mimetype):
    self.send_response(200)
    self.send_header('Content-type', mimetype)
    self.end_headers()

  def do_GET(self):
    if self.path == "/":
      self.path = "/index.html"

    if self.path == "/list":
      self._set_headers('text/html')
      self.wfile.write(result.render_list())
      return

    try:
      sendReply = False
      if self.path.endswith(".html"):
        mimetype = 'text/html'
        sendReply = True
      if self.path.endswith(".jpg"):
        mimetype = 'image/jpg'
        sendReply = True
      if self.path.endswith(".gif"):
        mimetype = 'image/gif'
        sendReply = True
      if self.path.endswith(".js"):
        mimetype = 'application/javascript'
        sendReply = True
      if self.path.endswith(".css"):
        mimetype = 'text/css'
        sendReply = True

      if sendReply == True:
        self._set_headers(mimetype)
        f = open(os.curdir + os.sep + self.path, 'rb')
        self.wfile.write(f.read())
        f.close()
      return

    except IOError:
      self.send_error(404, 'File Not Found: %s' % self.path)

  def do_POST(self):
    form = cgi.FieldStorage(fp=self.rfile,
                            headers=self.headers,
                            environ={'REQUEST_METHOD': 'POST'})
    image_content = form.value[0].value

    upload_dir = os.curdir + os.sep + 'uploads'
    filename =  '{}_{}.jpg'.format(str(len(os.listdir(upload_dir)) + 1), hashlib.md5(image_content).hexdigest())
    file_path =  upload_dir + os.sep + filename

    image = self._save_image(image_content, file_path)

    mlp_pred, mlp_prob = mlp.predict(image)
    shallow_pred, shallow_prob = shallow.predict(image)
    deep_pred, deep_prob = deep.predict(image)

    print('Filename: {}'.format(form.value[0].filename))
    print('MLP: {}'.format(mlp_pred))
    print('Shallow CNN: {}'.format(shallow_pred))
    print('Deep CNN: {}'.format(deep_pred))

    result_dir = os.curdir + os.sep + 'results'
    with open(result_dir + os.sep + filename + '.txt', 'w') as f:
      f.write(','.join(['mlp', mlp_pred, '{:.3f}'.format(mlp_prob)]) + '\n')
      f.write(','.join(['shallow', shallow_pred, '{:.3f}'.format(shallow_prob)]) + '\n')
      f.write(','.join(['deep', deep_pred, '{:.3f}'.format(deep_prob)]))

    self._set_headers('text/html')
    self.wfile.write(result.render_single(file_path, mlp_pred, mlp_prob, shallow_pred, shallow_prob, deep_pred, deep_prob))


  def _save_image(self, image_content, image_path):
    img = Image.open(io.BytesIO(image_content))
    img = self._center_crop(img)

    save_img = img.resize((256, 256), Image.ANTIALIAS)
    save_img.save(image_path)

    img = img.resize((48, 48), Image.ANTIALIAS)
    img = img.convert('L')  # convert image to grayscale
    img = np.asarray(img) / 255.0 - 0.5
    return img.reshape(1, 48, 48, 1)

  def _center_crop(self, img):
    width, height = img.size
    new_width = new_height = min(width, height)
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    return img.crop((left, top, right, bottom))




def run():
  server_address = ('127.0.0.1', 8429)
  httpd = HTTPServer(server_address, MyHandler)
  print('http server is running...')
  httpd.serve_forever()


if __name__ == '__main__':
  run()
