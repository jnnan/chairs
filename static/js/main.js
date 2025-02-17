var scripts = document.getElementsByTagName('script');
var myScript = scripts[scripts.length - 1];

var queryString = myScript.src.replace(/^[^\?]+\??/, '');

var params = parseQuery(queryString);

var recruit = 0;

function parseQuery(query) {
    var Params = {};
    if (!query) return Params; // return empty object
    var Pairs = query.split(/[;&]/);
    for (var i = 0; i < Pairs.length; i++) {
        var KeyVal = Pairs[i].split('=');
        if (!KeyVal || KeyVal.length != 2) continue;
        var key = unescape(KeyVal[0]);
        var val = unescape(KeyVal[1]);
        val = val.replace(/\+/g, ' ');
        Params[key] = val;
    }
    return Params;
}

function showPubs(id) {
  if (id == 0) {
    document.getElementById('pubs').innerHTML = document.getElementById('pubs_selected').innerHTML;
    document.getElementById('select0').style = 'font-weight:bold; background-color: #48C2EC';
    document.getElementById('select1').style = '';
    document.getElementById('select2').style = '';
  } else if (id == 1) {
    document.getElementById('pubs').innerHTML = document.getElementById('pubs_by_date').innerHTML;
    document.getElementById('select1').style = 'font-weight:bold; background-color: #48C2EC';
    document.getElementById('select0').style = '';
    document.getElementById('select2').style = '';
  } else {
    document.getElementById('pubs').innerHTML = document.getElementById('pubs_by_topic').innerHTML;
    document.getElementById('select2').style = 'font-weight:bold; background-color: #48C2EC';
    document.getElementById('select0').style = '';
    document.getElementById('select1').style = '';
  }
}

function showRecruit() {
  if (recruit == 0) {
    document.getElementById('recruit').style='display:inline-block';
  } else {
    document.getElementById('recruit').style='display:none';
  }
  recruit = 1 - recruit;
}