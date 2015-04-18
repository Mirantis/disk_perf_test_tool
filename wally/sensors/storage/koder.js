/* global _ */

/*
 * Complex scripted dashboard
 * This script generates a dashboard object that Grafana can load. It also takes a number of user
 * supplied URL parameters (int ARGS variable)
 *
 * Return a dashboard object, or a function
 *
 * For async scripts, return a function, this function must take a single callback function as argument,
 * call this callback function with the dashboard object (look at scripted_async.js for an example)
 */



// accessable variables in this scope
var window, document, ARGS, $, jQuery, moment, kbn;

// Setup some variables
var dashboard;

// All url parameters are available via the ARGS object
var ARGS;

// Intialize a skeleton with nothing but a rows array and service object
dashboard = {rows : []};

// Set a title
dashboard.title = 'Tests dash';

// Set default time
// time can be overriden in the url using from/to parameteres, but this is
// handled automatically in grafana core during dashboard initialization
dashboard.time = {
    from: "now-5m",
    to: "now"
};

dashboard.rows.push({
    title: 'Chart',
    height: '300px',
    panels: [{"span": 12, "title": "writes_completed", "linewidth": 2, "type": "graph", "targets": [{"alias": "192.168.0.104 io sda1", "interval": "", "target": "disk io", "rawQuery": true, "query": "select value from \"writes_completed\" where $timeFilter and host='192.168.0.104' and device='sda1' order asc"}, {"alias": "192.168.0.104 io rbd1", "interval": "", "target": "disk io", "rawQuery": true, "query": "select value from \"writes_completed\" where $timeFilter and host='192.168.0.104' and device='rbd1' order asc"}], "tooltip": {"shared": true}, "fill": 1}, {"span": 12, "title": "sectors_written", "linewidth": 2, "type": "graph", "targets": [{"alias": "192.168.0.104 io sda1", "interval": "", "target": "disk io", "rawQuery": true, "query": "select value from \"sectors_written\" where $timeFilter and host='192.168.0.104' and device='sda1' order asc"}, {"alias": "192.168.0.104 io rbd1", "interval": "", "target": "disk io", "rawQuery": true, "query": "select value from \"sectors_written\" where $timeFilter and host='192.168.0.104' and device='rbd1' order asc"}], "tooltip": {"shared": true}, "fill": 1}]
});


return dashboard;

