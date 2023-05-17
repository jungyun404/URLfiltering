// 현재 탭의 URL을 가져와 alert으로 표시하는 함수
chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    var activeTab = tabs[0];
    var tabUrl = activeTab.url;
  
    alert(tabUrl);
});
  
chrome.tabs.executeScript(function () {
    displayUrls();
  });