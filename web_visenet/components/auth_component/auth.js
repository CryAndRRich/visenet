(function(){
    window.flipTo = function(which){
      var fc = document.getElementById("flipCard");
      if(!fc) return;
      if(which === "back") fc.classList.add("flipped"); else fc.classList.remove("flipped");
    };
  
    window.togglePassword = function(id, el){
      var ip = document.getElementById(id);
      if(!ip) return;
      if(ip.type === "password"){ ip.type = "text"; el.textContent = "🙈"; }
      else { ip.type = "password"; el.textContent = "👁"; }
    };
  
    function postToServer(payload){
        fetch("http://localhost:8765/save_user", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        })
        .finally(function(){
            try { window.parent.location.reload(); } catch(e) { location.reload(); }
        });
    }
    
    window.submitLogin = function(){
        var u = document.getElementById("login_user").value || "";
        var p = document.getElementById("login_pass").value || "";
        postToServer({ type: "login", username: u, password: p });
    };
    window.submitRegister = function(){
        var u = document.getElementById("reg_user").value || "";
        var p = document.getElementById("reg_pass").value || "";
        postToServer({ type: "register", username: u, password: p });
    };
})();
  