package com.sparrowrecsys.online;

import com.sparrowrecsys.online.datamanager.DataManager;
import com.sparrowrecsys.online.service.*;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.servlet.DefaultServlet;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.eclipse.jetty.servlet.ServletHolder;
import org.eclipse.jetty.util.resource.Resource;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URL;

/***
 * Recsys Server, end point of online recommendation service
 */

public class RecSysServer {
    //主函数，创建推荐服务器并运行
    public static void main(String[] args) throws Exception {
        new RecSysServer().run();
    }

    //recsys server port number
    //推荐服务器的默认服务端口6010
    private static final int DEFAULT_PORT = 6010;

    //运行推荐服务器的函数
    public void run() throws Exception{

        int port = DEFAULT_PORT;
        try {
            port = Integer.parseInt(System.getenv("PORT"));
        } catch (NumberFormatException ignored) {}

        //set ip and port number
        //绑定IP地址和端口，0.0.0.0代表本地运行
        InetSocketAddress inetAddress = new InetSocketAddress("0.0.0.0", port);
        //创建Jetty服务器
        Server server = new Server(inetAddress);

        //get index.html path
        URL webRootLocation = this.getClass().getResource("/webroot/index.html");
        if (webRootLocation == null)
        {
            throw new IllegalStateException("Unable to determine webroot URL location");
        }

        //set index.html as the root page
        URI webRootUri = URI.create(webRootLocation.toURI().toASCIIString().replaceFirst("/index.html$","/"));
        System.out.printf("Web Root URI: %s%n", webRootUri.getPath());

        //load all the data to DataManager
        DataManager.getInstance().loadData(webRootUri.getPath() + "sampledata/movies.csv",
                webRootUri.getPath() + "sampledata/links.csv",webRootUri.getPath() + "sampledata/ratings.csv",
                webRootUri.getPath() + "modeldata/item2vecEmb.csv",
                webRootUri.getPath() + "modeldata/userEmb.csv",
                "i2vEmb", "uEmb");

        //create server context
        //创建Jetty服务器的环境handler
        ServletContextHandler context = new ServletContextHandler();
        context.setContextPath("/");
        context.setBaseResource(Resource.newResource(webRootUri));
        context.setWelcomeFiles(new String[] { "index.html" });
        context.getMimeTypes().addMimeMapping("txt","text/plain;charset=utf-8");

        //bind services with different servlets
        context.addServlet(DefaultServlet.class,"/");
        //添加API，getMovie，获取电影相关数据
        context.addServlet(new ServletHolder(new MovieService()), "/getmovie");
        //添加API，getuser，获取用户相关数据
        context.addServlet(new ServletHolder(new UserService()), "/getuser");
        //添加API，getsimilarmovie，获取相似电影推荐
        context.addServlet(new ServletHolder(new SimilarMovieService()), "/getsimilarmovie");
        //添加API，getrecommendation，获取各类电影推荐
        context.addServlet(new ServletHolder(new RecommendationService()), "/getrecommendation");
        //添加API，getrecforyou，为你推荐
        context.addServlet(new ServletHolder(new RecForYouService()), "/getrecforyou");

        //set url handler
        server.setHandler(context);
        System.out.println("RecSys Server has started.");

        //start Server
        //启动Jetty服务器
        server.start();
        server.join();
    }
}
