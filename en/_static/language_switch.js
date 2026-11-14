(function () {
  "use strict";

  try {
    localStorage.removeItem("relic-docs-theme");
  } catch (e) {}
  document.documentElement.removeAttribute("data-relic-theme");

  function languageTarget(currentUrl) {
    var target = new URL(currentUrl);
    var match = target.pathname.match(/\/(en|zh)\//);
    if (!match) {
      return null;
    }
    var currentLanguage = match[1];
    var nextLanguage = currentLanguage === "en" ? "zh" : "en";
    target.pathname = target.pathname.replace(
      "/" + currentLanguage + "/",
      "/" + nextLanguage + "/"
    );
    return {
      href: target.href,
      label: currentLanguage === "en" ? "中文" : "EN",
      portalLabel: currentLanguage === "en" ? "Home" : "首页",
      language: nextLanguage === "en" ? "en" : "zh-CN"
    };
  }

  window.RELICLanguageTarget = languageTarget;

  document.addEventListener("DOMContentLoaded", function () {
    var contentRoot = document.documentElement.getAttribute("data-content_root") || "./";

    var homeLink = document.querySelector(".wy-side-nav-search > a.icon-home");
    if (homeLink) {
      var logoStyle = "background:transparent!important;background-color:transparent!important;border-radius:0!important;padding:0!important;margin:0!important;height:46px!important;width:auto!important;display:inline-block!important;";
      var existing = homeLink.querySelector(".relic-sidebar-logo");
      if (existing) {
        existing.src = contentRoot + "_static/relic_mark_black.svg?v=mono5";
        existing.style.cssText = logoStyle;
      } else {
        var logo = document.createElement("img");
        logo.className = "relic-sidebar-logo";
        logo.src = contentRoot + "_static/relic_mark_black.svg?v=mono5";
        logo.alt = "RELIC";
        logo.style.cssText = logoStyle;
        homeLink.insertBefore(logo, homeLink.firstChild);
      }
    }

    var target = languageTarget(window.location.href);
    var bar = document.createElement("div");
    bar.className = "relic-utility-bar";
    bar.setAttribute("role", "region");
    bar.setAttribute("aria-label", "Language switch");

    var portalLink = document.createElement("a");
    portalLink.className = "relic-util-btn";
    portalLink.href = contentRoot + "../index.html";
    portalLink.textContent = target ? target.portalLabel : "Home";
    bar.appendChild(portalLink);

    if (target) {
      var link = document.createElement("a");
      link.className = "relic-language-switch";
      link.href = target.href;
      link.lang = target.language;
      link.textContent = target.label;
      link.setAttribute("aria-label", "Switch documentation language");
      bar.appendChild(link);
    }

    document.body.appendChild(bar);
  });
}());
