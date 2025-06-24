pipeline {
    agent any
    
    environment {
        DOMINO_API_KEY = credentials('domino-api-key')
        DOMINO_URL = 'https://se-demo.domino.tech'
        DOMINO_PROJECT = 'jim_coates/NLP_Quality_Analytics'
    }
    
    stages {
        stage('Debug Info') {
            steps {
                script {
                    echo "🔍 Debug Information:"
                    echo "Branch Name: ${env.BRANCH_NAME ?: 'NOT SET'}"
                    echo "Git Branch: ${env.GIT_BRANCH ?: 'NOT SET'}"
                    echo "Build Number: ${BUILD_NUMBER}"
                    echo "Domino URL: ${DOMINO_URL}"
                    echo "Domino Project: ${DOMINO_PROJECT}"
                    echo "API URL will be: ${DOMINO_URL}/v1/projects/${DOMINO_PROJECT}/runs"
                }
            }
        }
        
        stage('Trigger Domino Training Job') {
            steps {
                script {
                    echo "🚀 Starting Domino job using legacy API..."
                    
                    // Based on legacy API docs - more complete payload
                    def jobPayload = [
                        command: ["python", "experiments/mlflow_tracking.py"],
                        isDirect: false,
                        title: "Jenkins Triggered Training - Build ${BUILD_NUMBER}"
                        // Removed 'tier' for now - might not be required
                    ]
                    
                    echo "📤 Sending payload: ${groovy.json.JsonOutput.toJson(jobPayload)}"
                    
                    def response = httpRequest(
                        httpMode: 'POST',
                        url: "${DOMINO_URL}/v1/projects/${DOMINO_PROJECT}/runs",
                        customHeaders: [
                            [name: 'X-Domino-Api-Key', value: DOMINO_API_KEY, maskValue: true],
                            [name: 'Content-Type', value: 'application/json']
                        ],
                        requestBody: groovy.json.JsonOutput.toJson(jobPayload),
                        validResponseCodes: '200:299,400' // Allow 400 to see the error message
                    )
                    
                    echo "📨 Response status: ${response.status}"
                    echo "📄 Response body: ${response.content}"
                    
                    if (response.status == 200 || response.status == 201) {
                        def jobInfo = readJSON text: response.content
                        env.DOMINO_RUN_ID = jobInfo.runId
                        echo "✅ Started Domino run: ${jobInfo.runId}"
                        echo "🔗 View run at: ${jobInfo.message ?: 'Check your Domino project'}"
                    } else {
                        echo "❌ Failed to start run. Status: ${response.status}"
                        echo "❌ Error details: ${response.content}"
                        error("Failed to start Domino run")
                    }
                }
            }
        }
        
        stage('Monitor Training Progress') {
            when {
                expression { env.DOMINO_RUN_ID != null }
            }
            steps {
                script {
                    echo "⏳ Monitoring training progress..."
                    
                    timeout(time: 30, unit: 'MINUTES') {
                        waitUntil {
                            try {
                                // Based on legacy API: GET /v1/projects/{username}/{project_name}/runs/{run_id}
                                def statusResponse = httpRequest(
                                    httpMode: 'GET',
                                    url: "${DOMINO_URL}/v1/projects/${DOMINO_PROJECT}/runs/${env.DOMINO_RUN_ID}",
                                    customHeaders: [
                                        [name: 'X-Domino-Api-Key', value: DOMINO_API_KEY, maskValue: true]
                                    ],
                                    validResponseCodes: '200:299'
                                )
                                
                                def runStatus = readJSON text: statusResponse.content
                                def currentStatus = runStatus.status
                                echo "📊 Run status: ${currentStatus}"
                                
                                // Check if run is finished
                                return currentStatus in ['Succeeded', 'Failed', 'Error', 'Stopped']
                                
                            } catch (Exception e) {
                                echo "⚠️  Error checking run status: ${e.getMessage()}"
                                return true // Exit the wait loop on error
                            }
                        }
                    }
                }
            }
        }
        
        stage('Get Run Results') {
            when {
                expression { env.DOMINO_RUN_ID != null }
            }
            steps {
                script {
                    echo "📊 Fetching final run status..."
                    
                    try {
                        def finalResponse = httpRequest(
                            httpMode: 'GET',
                            url: "${DOMINO_URL}/v1/projects/${DOMINO_PROJECT}/runs/${env.DOMINO_RUN_ID}",
                            customHeaders: [
                                [name: 'X-Domino-Api-Key', value: DOMINO_API_KEY, maskValue: true]
                            ],
                            validResponseCodes: '200:299'
                        )
                        
                        def finalStatus = readJSON text: finalResponse.content
                        echo "🏁 Final run status: ${finalStatus.status}"
                        
                        if (finalStatus.status == 'Succeeded') {
                            echo "🎉 Training job completed successfully!"
                        } else {
                            echo "⚠️  Training job status: ${finalStatus.status}"
                            currentBuild.result = 'UNSTABLE'
                        }
                        
                    } catch (Exception e) {
                        echo "⚠️  Could not get final run status: ${e.getMessage()}"
                    }
                }
            }
        }
    }
    
    post {
        always {
            echo "🏁 Pipeline completed"
        }
        success {
            script {
                echo "✅ Successfully started and monitored Domino training job!"
                if (env.DOMINO_RUN_ID) {
                    echo "🎯 Run ID: ${env.DOMINO_RUN_ID}"
                    echo "🔗 Check run details at: ${DOMINO_URL}/projects/${DOMINO_PROJECT}/runs"
                }
            }
        }
        failure {
            echo "❌ Pipeline failed"
        }
        unstable {
            echo "⚠️  Pipeline completed with warnings"
        }
    }
}