import os

head = """
<!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta name="description" content="Caffe demos">
        <meta name="author" content="Preferred.AI (https://preferred.ai)">
  
        <title>Facial Emotion Recognition Demo</title>
  
        <link href="//netdna.bootstrapcdn.com/bootstrap/3.1.1/css/bootstrap.min.css" rel="stylesheet">
  
        <script type="text/javascript" src="//code.jquery.com/jquery-2.1.1.js"></script>
        <script src="//netdna.bootstrapcdn.com/bootstrap/3.1.1/js/bootstrap.min.js"></script>
  
            <!-- Script to instantly classify an image once it is uploaded. -->
        <script type="text/javascript">
          $(document).ready(
            function(){
              $('#imagefile').change(
                function(){
                  if ($(this).val()){
                    $('#formupload').submit();
                  }
                }
              );
            }
          );
        </script>
  
        <style>
        body {
          font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
          line-height:1.5em;
          color: #232323;
          -webkit-font-smoothing: antialiased;
        }
  
        h1, h2, h3 {
          font-family: Times, serif;
          line-height:1.5em;
          border-bottom: 1px solid #ccc;
        }
        </style>
      </head>
  
      <body>
        <!-- Begin page content -->
        <div class="container">
          <div class="page-header">
            <h1><a href="/">Facial Emotion Recognition</a></h1>
            <p>
              This is a demo of Facial Emotion Recognition Tutorial by <a href="https://preferred.ai/">Preferred.AI</a>.
            </p>
          </div>
"""

tail = """
        <hr>
        <div id="footer">
          <div class="container">
            <p>&copy; Preferred.AI</p>
          </div>
       </div>
     </body>
    </html>
"""


def render_upload(image_path, mlp_pred, mlp_prob, shallow_pred, shallow_prob, deep_pred, deep_prob):
  content = """
  
        <table style="width:70%">
          <tr>
            <th><img class="media-object" width="256" height="256" src="{}"></th>
            <th>
              <h4><strong>MLP:</strong> {} ({:.3f})</br></br></h4>
              <h4><strong>Shallow CNN:</strong> {} ({:.3f})</br></br></h4>
              <h4><strong>Deep CNN:</strong> {} ({:.3f})</br></br></h4>
            </th>
          </tr>
        </table>

          <hr>
      
          <form id="formupload" class="form-inline" role="form" action="" method="post" enctype="multipart/form-data">
            <div class="form-group">
              <label for="imagefile">Upload an image:</label>
              <input type="file" name="imagefile" id="imagefile">
            </div>
          </form>
        </div>
  """.format(image_path, mlp_pred, mlp_prob, shallow_pred, shallow_prob, deep_pred, deep_prob)

  return str.encode(head + content + tail)



result_map = {}

def render_list():
  global result_map

  content = """<table style="width:50%">"""

  upload_dir = os.curdir + os.sep + 'uploads'
  files = os.listdir(upload_dir)
  files.sort(key=lambda x: os.path.getmtime(upload_dir + os.sep + x))

  for file in files[::-1]:
    if file not in result_map:
      result_file = os.curdir + os.sep + 'results' + os.sep + file + '.txt'
      if not os.path.exists(result_file):
        continue
      with open(result_file, 'r') as f:
        img_result = {}
        for line in f:
          tokens = line.strip().split(',')
          img_result[tokens[0]] = {}
          img_result[tokens[0]]['pred'] = tokens[1]
          img_result[tokens[0]]['prob'] = tokens[2]
        result_map[file] = img_result

    image_path = upload_dir + os.sep + file
    mlp_pred = result_map[file]['mlp']['pred']
    mlp_prob = result_map[file]['mlp']['prob']
    shallow_prob = result_map[file]['shallow']['prob']
    shallow_pred = result_map[file]['shallow']['pred']
    deep_pred = result_map[file]['deep']['pred']
    deep_prob = result_map[file]['deep']['prob']

    content += """
        <tr>
          <th style="padding-top: 10px; padding-bottom: 10px"><img class="media-object" width="192" height="192" src="{}"></th>
          <th>
            <h4><strong>MLP:</strong> {} ({})</br></br></h4>
            <h4><strong>Shallow CNN:</strong> {} ({})</br></br></h4>
            <h4><strong>Deep CNN:</strong> {} ({})</br></br></h4>
          </th>
        </tr>
    """.format(image_path, mlp_pred, mlp_prob, shallow_pred, shallow_prob, deep_pred, deep_prob)

  content += """</table>"""

  return str.encode(head + content + tail)