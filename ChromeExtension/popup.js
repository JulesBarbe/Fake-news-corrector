btn = document.getElementById("checkPage");
btn.onclick = function() {
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        var activeTab = tabs[0];
        var url = activeTab.url;
        window.open("http://localhost:5000/external?url=" + url)
    });
}