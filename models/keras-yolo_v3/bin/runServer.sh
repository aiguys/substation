#!/usr/bin/env bash
export LANG=zh_CN.UTF-8

PRG_KEY="yolo3"
RUN_PATH="/Users/henry/Documents/application/keras-yolo3/"


cd $RUN_PATH

case "$1" in
    start)
        nohup python3 -u $RUN_PATH/server.py runserver > nohup.log 2>&1 &

        echo "$PRG_KEY started, please check log."

        ;;

    stop)
        pid=$(pgrep -f $PRG_KEY)
        echo "$pid"
        kill -9 $pid
            ##killall python3
        echo "$PRG_KEY stoped!"

        ;;

    restart)
        $0 stop
        sleep 1
        $0 start

        ;;
    *)
        echo "Usage: $0 {start|stop|restart}"
        exit 1
esac

exit 0
